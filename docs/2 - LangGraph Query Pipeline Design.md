# LangGraph Query Pipeline Design

## Overview

The query pipeline is a LangGraph-based multi-step orchestration system that handles two categories of evaluator queries: action item retrieval and evaluation note lookup. Queries outside these two categories are rejected at the router and never enter the graph.

The pipeline is stateful across turns using a DynamoDB-backed checkpointer. Every invocation restores the full graph state from the previous turn, enabling follow-up questions without re-sending context from the frontend.

All retrieval is performed by MCP Lambda servers registered behind the AWS Bedrock AgentCore Gateway. LangGraph nodes invoke these servers via the MCP protocol. The Lambda servers are thin I/O adapters: they execute queries against DynamoDB or OpenSearch and return raw results. Orchestration logic (retry, fallback, MMR reranking) lives inside LangGraph nodes, not inside the Lambda.

---

## High-Level Flow

```
React Frontend
    │
    ▼
FastAPI Backend
    │  (query, device_id, thread_id)
    ▼
LangGraph Graph (thread resumed via DynamoDBSaver)
    │
    ├── Router Node
    │       ├── Out-of-scope → return message directly, graph terminates
    │       ├── action_item intent → Validation Node
    │       └── evaluation_note intent → Validation Node
    │
    ├── Validation Node
    │       ├── Validation failure (up to 3 retries) → hard error exit
    │       └── Validation pass
    │               ├── action_item path → Device Lookup Node → Action Item Node → Summarization Node → Generation Node
    │               └── evaluation_note path → Evaluation Note Node → Summarization Node → Generation Node
    │
    └── Generation Node
            │  (StreamWriter emits tokens to FastAPI)
            ▼
        RAGAS Evaluation Node (async, post-generation)
            │
            ▼
        CloudWatch (SecurityEvalRAG namespace)
```

---

## Graph State

The state schema is a `TypedDict` shared across all nodes. DynamoDBSaver snapshots the entire object after every node execution and rehydrates it in full before the next invocation on the same thread.

```python
class AgentState(TypedDict):
    messages:             Annotated[list, add_messages]   # Full conversation history, append-only
    conversation_history: list                            # Trimmed window, managed by summarization node
    summary:              str                             # Rolling LLM-generated summary of older turns
    intent:               str                             # "action_item" | "evaluation_note" | "out_of_scope"
    device_id:            str                             # Extracted from query or inherited from prior turn
    device_context:       dict                            # Full device record from DynamoDB (device_type, manufacturer, etc.)
    retrieved_chunks:     list                            # Post-MMR chunk list from OpenSearch (evaluation_note path)
    action_items:         list                            # Action item records from DynamoDB (action_item path)
    context_present:      bool                            # False if retrieval returned empty results
    rejection_note:       str                             # Error or out-of-scope message to return to user
    generation_output:    str                             # Final LLM-generated answer
```

Key design decisions:

`messages` uses the `add_messages` reducer so each new user and assistant message appends to the list rather than replacing prior turns. This is the field the router reads to resolve ambiguous follow-up queries.

`conversation_history` is a separate, managed copy of the recent message window. The summarization node truncates this to the last 10 messages and extends `summary`. The generation node reads `conversation_history` and `summary` to assemble the LLM prompt context block.

`context_present` is a boolean flag written by the retrieval nodes. The generation node reads this flag to decide whether to invoke the LLM or return the canned empty-context message directly. This prevents the LLM from fabricating responses when no grounded context exists.

---

## Node Definitions

### 1. Router Node

**Responsibility:** Classify the intent of the incoming query against the last 5 turns of conversation history, extract device ID if present, and either return an out-of-scope message or pass to the validation node with intent written to state.

**Inputs from state:** `messages` (last 5 entries), `device_id` (may be inherited from prior turn)

**Outputs to state:** `intent`, `device_id`, `rejection_note` (if out-of-scope)

**Behavior:**

The router reads the last 5 entries from `state["messages"]` and includes them in the classification prompt alongside the current query. This allows follow-up questions like "what about the firmware version findings?" to be correctly resolved as continuations of a prior evaluation note query rather than treated as ambiguous standalone queries.

The router classifies into exactly three categories:

- `action_item`: the evaluator is asking what checks or tasks apply to a specific device
- `evaluation_note`: the evaluator is asking about past findings, incidents, or compliance observations for a device
- `out_of_scope`: anything else

If `out_of_scope`, the router writes a rejection message to `rejection_note` and the graph terminates. No downstream nodes execute.

Device ID is extracted from the query text if present. If not present in the current query, the prior value of `device_id` in state (restored by DynamoDBSaver) is inherited. This is what allows multi-turn conversations to reference the same device without restating it.

**Graph edge:** out-of-scope terminates; action_item or evaluation_note proceeds to the validation node.

---

### 2. Validation Node

**Responsibility:** Validate that a device ID is present in state and that the device record exists in DynamoDB. Write the resolved device record to `device_context` in state.

**Inputs from state:** `device_id`, `intent`

**Outputs to state:** `device_context`, `rejection_note` (on failure)

**Behavior:**

Two sequential checks:

1. Structural check: is `device_id` a non-empty value in state. If not, write a validation error to `rejection_note` and terminate. This is a caller error; retry cannot fix it.

2. Existence check: call the device lookup MCP Lambda via the Bedrock AgentCore Gateway. If the device record is not found, write an error to `rejection_note` and terminate.

On success, write the full device record (device_type, manufacturer, firmware_version, category, etc.) to `device_context`. This resolves the sort key problem: downstream nodes read `device_type` from `device_context` rather than querying for it independently. The data dependency is satisfied by the time the graph leaves this node.

Validation failure is not retried because it is not a transient error. The router retry loop (up to 3 attempts) applies only if the validation node determines the payload was malformed due to a router classification error, not a missing device.

**Graph edge:** failure terminates with error; success proceeds to either the device lookup node (action item path) or the evaluation note node (evaluation note path) based on `intent`.

---

### 3. Device Lookup Node (Action Item Path Only)

**Responsibility:** Fetch the full device record from DynamoDB via MCP Lambda to confirm device type is available in state before action item retrieval.

**Note:** In most cases this work is already done by the validation node. This node exists as an explicit guard to enforce the data dependency in the graph topology. It ensures `device_context` is always populated before the action item node runs, making the ordering a graph-level guarantee rather than a runtime assumption.

**Inputs from state:** `device_id`, `device_context`

**Outputs to state:** `device_context` (refreshed if needed)

**Graph edge:** proceeds to the action item node.

---

### 4. Action Item Node (Action Item Path Only)

**Responsibility:** Fetch action items from DynamoDB for the resolved device type via MCP Lambda.

**Inputs from state:** `device_context` (specifically `device_type`)

**Outputs to state:** `action_items`, `context_present`

**Behavior:**

Calls the action item MCP Lambda with `device_type` as the query parameter. The Lambda performs a DynamoDB query using `device_type` as the partition key (or via a GSI if the data model requires it) and returns the list of action items.

Retry: up to 3 attempts with exponential backoff on MCP Gateway timeout. After 3 failures, write an error to `rejection_note` and terminate the graph.

If the Lambda returns an empty list, set `context_present = False`. The generation node will return the empty-context message rather than calling the LLM.

If the Lambda returns results, set `context_present = True` and write results to `action_items`.

**Graph edge:** proceeds to the summarization node.

---

### 5. Evaluation Note Node (Evaluation Note Path Only)

**Responsibility:** Retrieve semantically relevant evaluation notes from OpenSearch via MCP Lambda, apply MMR reranking, and write the final chunk set to state.

**Inputs from state:** `device_context`, `messages` (current query)

**Outputs to state:** `retrieved_chunks`, `context_present`

**Behavior:**

Calls the evaluation note MCP Lambda with the current query text and relevant metadata filters (device_type, manufacturer, facility_id if applicable). The Lambda executes a hybrid search against OpenSearch combining dense vector similarity (Titan embeddings on `findings_embedding`) with BM25 lexical matching, scoped by metadata filter clauses.

The Lambda returns the top K candidate chunks. MMR reranking runs inside the Lambda on this candidate set using the `findings_embedding` vectors stored in OpenSearch. The post-MMR top N chunks are returned to the LangGraph node.

The node writes the resulting chunk list to `retrieved_chunks`. If the list is empty, `context_present = False`. The generation node bypasses the LLM and returns the canned message: no prior evaluations match this scenario; the evaluator should treat this as a novel case requiring manual assessment.

Retry: up to 3 attempts with exponential backoff on MCP Gateway timeout. After 3 failures, write an error to `rejection_note` and terminate.

**Graph edge:** proceeds to the summarization node.

---

### 6. Summarization Node

**Responsibility:** Compress conversation history before the generation node assembles the LLM prompt. This node is the single convergence point for both paths.

**Inputs from state:** `messages`, `conversation_history`, `summary`

**Outputs to state:** `conversation_history` (truncated to last 10), `summary` (extended)

**Behavior:**

Reads the full `messages` list from state. Compares against the existing `conversation_history` window to identify messages that are new since the last invocation. If `conversation_history` would exceed 10 messages after appending the new ones, the node calls an LLM to extend the rolling `summary` with the oldest messages being evicted from the window, then truncates `conversation_history` to the last 10 messages.

If the history is within the 10-message window, the node appends the new messages and returns without calling the LLM.

The result is that the generation node always receives a bounded `conversation_history` (at most 10 messages) plus a `summary` string covering all older context. This keeps LLM prompt size predictable regardless of session length.

**Graph edge:** proceeds to the generation node on both paths.

---

### 7. Generation Node

**Responsibility:** Assemble the LLM prompt from state, call Bedrock, stream the response to the FastAPI caller via StreamWriter, and write the final answer to state.

**Inputs from state:** `intent`, `context_present`, `retrieved_chunks` or `action_items`, `device_context`, `conversation_history`, `summary`, `messages` (current query)

**Outputs to state:** `generation_output`, appends assistant message to `messages`

**Behavior:**

First checks `context_present`. If `False`, returns the canned fallback message immediately without calling the LLM.

If `True`, assembles the prompt using the following template structure:

```
<System>
You are a security evaluation assistant. Your answers must be grounded strictly
in the provided context. Do not draw on general knowledge or infer beyond what
the context supports.
</System>

<Conversations>
[summary field if non-empty]
[conversation_history last 10 messages]
</Conversations>

<DeviceContext>
device_id: {device_id}
device_type: {device_context.device_type}
manufacturer: {device_context.manufacturer}
</DeviceContext>

<RetrievedContext>
[retrieved_chunks as structured text, one chunk per block]
OR
[action_items as structured list]
</RetrievedContext>

<Query>
{current user query from messages}
</Query>
```

The LLM call goes to Bedrock. The node uses `StreamWriter` to emit tokens to the FastAPI SSE stream in real time as they arrive, so the evaluator sees the response progressively rather than waiting for full generation.

After generation completes, the node writes the full generated text to `generation_output` and appends it as an assistant message to `messages`.

**Graph edge:** proceeds to the RAGAS evaluation node.

---

### 8. RAGAS Evaluation Node

**Responsibility:** Compute RAG quality metrics and emit them to CloudWatch asynchronously after the response has been streamed to the user.

**Inputs from state:** `messages` (current query), `retrieved_chunks` or `action_items`, `generation_output`

**Outputs:** CloudWatch metrics (no state mutation)

**Behavior:**

This node runs after the generation node and after the response has already been streamed to the user. It does not block the user-facing response path.

Constructs a RAGAS `EvaluationDataset` sample from the three required fields: query, context, and answer. Runs the following metrics:

- `Faithfulness`: is the generated answer supported by the retrieved context
- `AnswerRelevancy`: does the answer address the query
- `ContextPrecision`: how much of the retrieved context contributed to the answer

Emits each metric as a CloudWatch custom metric under the `SecurityEvalRAG` namespace. Each metric is dimensioned by `device_type` so retrieval quality can be tracked per device category over time.

The RAGAS evaluation adds latency but this is non-blocking from the user's perspective since the response is already streamed. If the evaluation call fails, it is logged and dropped silently. It must never block or degrade the user-facing response.

---

## Context Management

Session context is managed at two levels.

**DynamoDBSaver (cross-turn persistence):** The `langgraph-checkpoint-aws` package provides `DynamoDBSaver` as the graph checkpointer. It is passed at `graph.compile()` time. On every node execution, LangGraph checkpoints the full `AgentState` TypedDict to DynamoDB keyed by `thread_id`. On the next invocation, the full state is rehydrated before any node runs. No explicit fetch is needed inside nodes.

**Summarization node (within-session compression):** The `conversation_history` field is bounded to the last 10 messages. Older turns are compressed into the `summary` field by an LLM call inside the summarization node. This keeps the generation node's prompt context size predictable. The rolling summary accumulates across turns and is persisted in DynamoDB as part of the checkpoint.

**Router context window (intent classification):** The router reads only the last 5 entries from `messages` when building its classification prompt. This is sufficient for follow-up resolution and keeps the router call cheap.

---

## Document Retrieval Design (Evaluation Note Path)

Retrieval uses a two-step pattern: hybrid search in OpenSearch followed by MMR reranking, both executed inside the evaluation note MCP Lambda.

**Step 1: Hybrid Search**

OpenSearch executes a combined query of two components: a `knn` vector query against the `findings_embedding` field using the Titan-embedded query vector, and a BM25 `match` query against the `findings_text` field for lexical signal. Results from both are fused using Reciprocal Rank Fusion (RRF). Metadata pre-filters scope the candidate set by `device_type`, `manufacturer`, and optionally `facility_id` before scoring runs. The top K candidates (typically 20) are returned.

**Step 2: MMR Reranking**

MMR runs on the K candidates in the Lambda. It uses the stored `findings_embedding` vectors (indexed with `store: true` in OpenSearch) to compute pairwise similarity across the candidate set. The lambda parameter controls the tradeoff between relevance to the query and diversity within the result set. The top N post-MMR chunks (typically 5) are returned to the LangGraph node.

**Empty result handling:** If the post-MMR set is empty, the node sets `context_present = False` and the generation node returns the canned no-context message. The LLM is never called without grounded context.

---

## Observability

| Signal | Mechanism | Details |
|---|---|---|
| RAG quality metrics | RAGAS + CloudWatch | Faithfulness, AnswerRelevancy, ContextPrecision emitted to `SecurityEvalRAG` namespace, dimensioned by `device_type` |
| MCP Gateway call latency and errors | CloudWatch (AgentCore runtime metrics) | Invocation count, throttled requests, error rate, end-to-end latency per endpoint |
| Node-level retry events | CloudWatch Logs | Each retry attempt inside a node logs the attempt count and error type |
| Graph termination on error | CloudWatch Logs | Rejection note written to state is also emitted as a log event with thread ID and intent |
| Token streaming | FastAPI SSE | Tokens streamed progressively to frontend via StreamWriter; no separate metric needed |
| Checkpoint writes | DynamoDB | Each successful node execution writes a checkpoint; DynamoDB CloudWatch metrics cover write latency and throttling |

---

## Error Handling Summary

| Condition | Handling |
|---|---|
| Out-of-scope query | Router writes `rejection_note`, graph terminates immediately |
| Device ID missing from query and state | Validation node writes error to `rejection_note`, graph terminates |
| Device not found in DynamoDB | Validation node writes error to `rejection_note`, graph terminates |
| MCP Gateway timeout | Node retries up to 3 times with exponential backoff; after 3 failures writes `rejection_note` and terminates |
| OpenSearch empty result | `context_present = False`; generation node returns canned message, no LLM call |
| Validation schema failure (router malformed payload) | Validation retries router up to 3 times; after 3 failures terminates with hard error |
| RAGAS evaluation failure | Logged and dropped silently; does not affect user-facing response |****