# LangGraph Query Pipeline Design

## Overview

The query pipeline is a LangGraph-based multi-step orchestration system that handles two categories of evaluator queries: action item retrieval and evaluation note lookup. Queries outside these two categories are rejected at the intent classifier node and never enter the rest of the graph.

The pipeline is stateful across turns using a DynamoDB-backed checkpointer (`DynamoDBSaver` from `langgraph-checkpoint-aws`). Every invocation rehydrates the full graph state from the previous turn before any node executes, enabling follow-up questions without re-sending context from the frontend.

All retrieval is performed by MCP Lambda servers registered behind the AWS Bedrock AgentCore Gateway. LangGraph nodes invoke these servers via the MCP protocol. Lambda servers are thin I/O adapters: they execute queries against DynamoDB or OpenSearch and return raw results. Retry logic, fallback behavior, and state management live inside LangGraph nodes, not inside the Lambda.

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
    ├── Intent Classifier Node
    │       ├── out_of_scope    → return message directly, graph terminates
    │       ├── action_item     → Validation Node
    │       └── evaluation_note → Validation Node
    │
    ├── Validation Node
    │       ├── Failure (up to 3 retries back to intent classifier) → hard error exit
    │       └── Pass
    │               ├── action_item path    → Device Lookup Node → Action Item Node → Summarization Node → Generation Node → RAGAS Evaluation Node
    │               └── evaluation_note path → Entity Extraction Node → Evaluation Note Node → Summarization Node → Generation Node → RAGAS Evaluation Node
```

---

## Graph State

The state schema is a `TypedDict` shared across all nodes. `DynamoDBSaver` snapshots the entire object after every node execution and rehydrates all fields before the next invocation on the same thread. Nodes never need to fetch their own prior state explicitly.

```python
class AgentState(TypedDict):
    messages:          Annotated[list, add_messages]  # Full conversation history, append-only reducer
    recent_messages:   list                           # Bounded window of last 10 messages, managed by summarization node
    summary:           str                            # Rolling LLM-generated summary of turns older than the window
    intent:            str                            # "action_item" | "evaluation_note" | "out_of_scope"
    device_id:         str                            # Extracted by intent classifier (action_item path only)
    device_context:    dict                           # Full device record resolved by device lookup node (action_item path only)
    device_type:       Optional[str]                  # Canonical value from normalized dictionary (evaluation_note path only)
    location:          Optional[str]                  # Canonical value from normalized dictionary (evaluation_note path only)
    retrieved_chunks:  list                           # Post-MMR chunks from OpenSearch (evaluation_note path)
    action_items:      list                           # Action item records from DynamoDB (action_item path)
    rejection_note:    str                            # Error or out-of-scope message returned to user on any exit
```

Key design decisions:

`messages` uses the `add_messages` reducer so each new user and assistant message appends rather than replaces. This is the accumulating source of truth for the full conversation across all turns and is persisted by `DynamoDBSaver` between sessions.

`recent_messages` is a separate bounded copy managed exclusively by the summarization node. It holds at most the last 10 messages. The generation node reads this field, not `messages`, when assembling the LLM prompt context block.

`intent` is written on the first turn and restored by `DynamoDBSaver` on subsequent turns. The intent classifier reads the restored value on follow-up queries and only re-classifies when the current query signals a topic shift, keeping the classification call cheap for the common multi-turn case.

`device_type` and `location` are optional fields present only on the evaluation note path. They are written by the entity extraction node only when the normalized dictionary lookup succeeds. If lookup fails for a field, that field is deleted from state. The evaluation note node reads whatever is present and builds the optional OpenSearch filter accordingly.

---

## Node Definitions

### 1. Intent Classifier Node

**Responsibility:** Classify the intent of the incoming query, extract relevant identifiers, and route to the next node. Return a rejection message and terminate immediately for out-of-scope queries.

**Inputs from state:** `messages` (current query), `intent` (restored from prior turn if present)

**Outputs to state:** `intent`, `device_id` (action_item path only), raw `device_type` and raw `location` strings (evaluation_note path, pre-normalization), `rejection_note` (out-of-scope only)

**Behavior:**

On the first turn of a session, `intent` is not yet in state. The node runs full classification using the current query text.

On follow-up turns, `intent` is restored by `DynamoDBSaver`. The node reads the restored `intent` as the anchor and checks whether the current query is a continuation or a topic shift. If it is a continuation, the restored intent is kept and no re-classification occurs. If the query clearly signals a new topic, the node re-classifies. This keeps the intent classifier call cheap for the common case of multi-turn follow-up within the same session.

Classification resolves to one of three values:

`action_item`: the evaluator is asking what checks or tasks apply to a specific device. The node extracts `device_id` from the query text and writes it to state. If `device_id` is absent from the current query, the prior value in state restored by `DynamoDBSaver` is inherited.

`evaluation_note`: the evaluator is asking about past findings, incidents, or compliance observations. The node attempts to extract raw device type and location strings only if they are explicitly stated in the query text. These are raw unresolved strings at this stage. Neither field is required and inference from ambiguous language is not performed here.

`out_of_scope`: anything that does not map to the above two categories. The node writes a rejection message to `rejection_note` and the graph terminates immediately.

**Graph edge:** `out_of_scope` terminates; `action_item` and `evaluation_note` proceed to the validation node.

---

### 2. Validation Node

**Responsibility:** Validate that required fields are present in state before the graph proceeds to retrieval.

**Inputs from state:** `intent`, `device_id`

**Outputs to state:** `rejection_note` (on failure)

**Behavior:**

On the `action_item` path: checks that `device_id` is a non-empty value in state. If absent, this is a caller error that retry cannot fix. The node writes an error to `rejection_note` and terminates the graph immediately.

On the `evaluation_note` path: no required fields exist since `device_type` and `location` are both optional. The validation node is a pass-through on this path.

The retry loop (up to 3 attempts back to the intent classifier) applies only when the validation node determines the payload was structurally malformed due to a classification ambiguity. A missing `device_id` is not retried since it is a caller error, not a transient classification failure.

**Graph edge:** failure terminates; pass proceeds to entity extraction (evaluation_note path) or device lookup (action_item path).

---

### 3. Entity Extraction Node (Evaluation Note Path Only)

**Responsibility:** Resolve raw device type and location strings extracted by the intent classifier into canonical values using a normalized dictionary fetched from DynamoDB via MCP Lambda. Delete any field from state that cannot be resolved.

**Inputs from state:** raw `device_type` string (optional), raw `location` string (optional)

**Outputs to state:** canonical `device_type` (optional), canonical `location` (optional)

**Behavior:**

Calls the normalization dictionary MCP Lambda via the Bedrock AgentCore Gateway to fetch the current dictionary from DynamoDB. The dictionary is a list of canonical string values covering known device types and location names.

Retry: up to 3 attempts with exponential backoff on MCP Gateway timeout. After 3 failures, writes an error to `rejection_note` and terminates the graph.

After fetching the dictionary, performs a local lookup for each extracted field:

For `device_type`: if the raw string matches a canonical value in the dictionary (case-insensitive), writes the canonical value to state. If no match is found, deletes `device_type` from state entirely.

For `location`: same lookup logic. If no match is found, deletes `location` from state entirely.

The result is that by the time this node completes, `device_type` and `location` in state are either absent or guaranteed to be valid canonical values. The evaluation note node never receives an unresolved or ambiguous filter value.

**Graph edge:** proceeds to the evaluation note node.

---

### 4. Device Lookup Node (Action Item Path Only)

**Responsibility:** Fetch the full device record from DynamoDB via MCP Lambda using the validated device ID. Return an error if the device does not exist.

**Inputs from state:** `device_id`

**Outputs to state:** `device_context`, `rejection_note` (on not-found or failure)

**Behavior:**

Calls the device lookup MCP Lambda with `device_id`. The Lambda performs a DynamoDB point lookup and returns the full device record including `device_type`, `manufacturer`, `firmware_version`, and `category`.

Retry: up to 3 attempts with exponential backoff on MCP Gateway timeout. After 3 failures, writes an error to `rejection_note` and terminates.

If the Lambda returns a not-found response, writes a user-facing error to `rejection_note` and terminates. This is not a transient error and is not retried.

On success, writes the full device record to `device_context`. All downstream nodes on the action item path read `device_type` from `device_context`. This guarantees `device_type` is present in state before the action item node runs, enforced by the graph topology rather than by LLM reasoning.

**Graph edge:** failure or not-found terminates; success proceeds to the action item node.

---

### 5. Action Item Node (Action Item Path Only)

**Responsibility:** Fetch action items from DynamoDB for the resolved device type via MCP Lambda.

**Inputs from state:** `device_context` (specifically `device_type`)

**Outputs to state:** `action_items`, `rejection_note` (on empty result or failure)

**Behavior:**

Calls the action item MCP Lambda with `device_type` read from `device_context`. The Lambda queries DynamoDB and returns the list of action items for that device type.

Retry: up to 3 attempts with exponential backoff on MCP Gateway timeout. After 3 failures, writes an error to `rejection_note` and terminates.

If the Lambda returns an empty list, writes a user-facing message to `rejection_note` and terminates immediately. The generation node is not called. The message informs the evaluator that no action items are defined for this device type.

On success, writes the action item list to `action_items` and proceeds.

**Graph edge:** empty result or failure terminates; success proceeds to the summarization node.

---

### 6. Evaluation Note Node (Evaluation Note Path Only)

**Responsibility:** Retrieve semantically relevant evaluation notes from OpenSearch via MCP Lambda using native MMR reranking.

**Inputs from state:** `messages` (current query text), `device_type` (optional canonical value), `location` (optional canonical value)

**Outputs to state:** `retrieved_chunks`, `rejection_note` (on empty result or failure)

**Behavior:**

Calls the evaluation note MCP Lambda with the current query text and any optional filter values present in state. The Lambda constructs and executes an OpenSearch knn query with native MMR reranking.

**Retrieval design inside the Lambda:**

The query is a top-level `knn` query against the `findings_embedding` field using the Titan-embedded query vector. MMR is applied natively via the `ext.mmr` parameter on the search request. The `diversity` parameter controls the relevance-diversity tradeoff. OpenSearch oversamples candidates internally, applies MMR reranking, and returns the final post-MMR top N results.

BM25 hybrid search is not used on this path. OpenSearch native MMR requires a `knn` or `neural` query as the top-level query and is explicitly incompatible with hybrid or bool query wrappers due to limitations in the hybrid search executor.

If `device_type` is present in state, it is applied as a metadata pre-filter on the knn query scoping the candidate set to matching documents. If `location` is present, it is applied as an additional pre-filter. Metadata filters are a separate mechanism from the query type and are fully compatible with knn queries and native MMR. If neither field is in state, the query runs unfiltered against the full index.

Retry: up to 3 attempts with exponential backoff on MCP Gateway timeout. After 3 failures, writes an error to `rejection_note` and terminates.

If the post-MMR result set is empty, writes a user-facing message to `rejection_note` and terminates immediately. The message informs the evaluator that no prior evaluations match this scenario and the case should be treated as novel requiring manual assessment. The LLM is never called without grounded context.

On success, writes the post-MMR chunk list to `retrieved_chunks` and proceeds.

**Graph edge:** empty result or failure terminates; success proceeds to the summarization node.

---

### 7. Summarization Node

**Responsibility:** Compress conversation history before the generation node assembles the LLM prompt. Sits as the single convergence point for both retrieval paths immediately before generation.

**Inputs from state:** `messages`, `recent_messages`, `summary`

**Outputs to state:** `recent_messages` (updated window), `summary` (extended if eviction occurred)

**Behavior:**

This is a custom LangGraph node with no external library dependency. The only message types in this system are `HumanMessage` and `AIMessage`, since the graph uses explicit sequential nodes rather than a tool-calling loop. There are no `ToolMessage` blocks to handle.

On each invocation the node compares the full `messages` list against the current `recent_messages` window to identify newly accumulated messages since the last invocation. It appends new messages to `recent_messages`.

If `recent_messages` would exceed 10 messages after appending, the oldest messages beyond the window are evicted. Before eviction, the node calls the Bedrock LLM to extend the rolling `summary` by incorporating the content of the messages being evicted. The `summary` field accumulates across turns and is persisted in DynamoDB as part of the checkpoint.

If `recent_messages` is within the 10-message limit after appending, no LLM call is made and the node writes the updated window back to state directly.

The generation node always receives a bounded `recent_messages` list of at most 10 messages plus a `summary` string covering all older context. Prompt size is predictable regardless of session length.

**Graph edge:** proceeds to the generation node on both paths.

---

### 8. Generation Node

**Responsibility:** Assemble the LLM prompt, call Bedrock, stream the response to the FastAPI caller via StreamWriter, and append the assistant message to state.

**Inputs from state:** `intent`, `retrieved_chunks` or `action_items`, `device_context` (action_item path), `device_type` and `location` (evaluation_note path, if present), `recent_messages`, `summary`, `messages` (current query)

**Outputs to state:** appends assistant `AIMessage` to `messages`

**Behavior:**

Because the action item node and evaluation note node both perform early exits on empty results, the generation node only runs when valid retrieval results are present in state. No additional empty-check is needed inside this node.

Assembles the prompt using the following structure:

```
<s>
You are a security evaluation assistant. Your answers must be grounded strictly
in the provided context. Do not draw on general knowledge or infer beyond what
the context supports.
</s>

<Conversations>
{summary if non-empty}
{recent_messages}
</Conversations>

<DeviceContext>
{device_id, device_type, manufacturer from device_context  -- action_item path}
{device_type, location if present                          -- evaluation_note path}
</DeviceContext>

<RetrievedContext>
{retrieved_chunks as structured text, one chunk per block  -- evaluation_note path}
{action_items as structured list                           -- action_item path}
</RetrievedContext>

<Query>
{current user query}
</Query>
```

The LLM call goes to Bedrock. The node uses `StreamWriter` to emit tokens to the FastAPI SSE stream in real time as they arrive. The evaluator sees the response progressively without waiting for full generation to complete.

After generation completes, the full generated text is appended as an `AIMessage` to `messages`.

**Graph edge:** proceeds to the RAGAS evaluation node.

---

### 9. RAGAS Evaluation Node

**Responsibility:** Compute RAG quality metrics from the completed turn and emit them to CloudWatch. Non-blocking from the user's perspective since the response is already fully streamed before this node runs.

**Inputs from state:** `messages` (to extract the current query and last assistant message), `retrieved_chunks` or `action_items`, `device_context` (action_item path) or `device_type` (evaluation_note path, if present)

**Outputs:** CloudWatch metrics only. No state mutation.

**Conceptual flow:**

The node constructs an evaluation sample from three fields available in state: the current user query (last `HumanMessage` in `messages`), the retrieved context (`retrieved_chunks` or `action_items`), and the generated answer (last `AIMessage` in `messages`).

This sample is passed to the RAGAS `evaluate()` function configured with a Bedrock LLM as the judge and three metrics:

`Faithfulness`: measures whether every claim in the generated answer is supported by the retrieved context. The LLM-as-judge decomposes the answer into atomic claims and verifies each one against the context chunks.

`AnswerRelevancy`: measures how directly the generated answer addresses the user query. The LLM-as-judge scores alignment between query and answer.

`ContextPrecision`: measures how much of the retrieved context actually contributed to the answer. Penalises over-retrieval where chunks were fetched but not used.

Each metric produces a float score between 0 and 1. The node emits each score as a CloudWatch custom metric under the `SecurityEvalRAG` namespace via `boto3.put_metric_data`. Each metric is dimensioned by `device_type` where available, resolved from `device_context` on the action item path or from `device_type` in state on the evaluation note path. This allows retrieval quality to be tracked per device category over time.

If the RAGAS evaluation call fails for any reason, the failure is logged and silently dropped. It must never affect the user-facing response path.

---

## Context Management

Session context is managed at two levels.

**DynamoDBSaver (cross-session persistence):** `DynamoDBSaver` from `langgraph-checkpoint-aws` is passed as the `checkpointer` argument at `graph.compile()` time. On every node execution, LangGraph checkpoints the full `AgentState` TypedDict to DynamoDB keyed by `thread_id`. On the next invocation with the same `thread_id`, all fields are rehydrated before any node runs. Nodes read prior state as if it were always present. This includes `messages`, `recent_messages`, `summary`, `intent`, and all other fields, enabling both within-session and cross-session continuity.

**Summarization node (within-session compression):** `recent_messages` is bounded to the last 10 messages. Older turns are compressed into the `summary` field via a Bedrock LLM call inside the summarization node. The rolling summary accumulates across turns and is persisted in DynamoDB as part of the checkpoint. The generation node assembles the prompt from `recent_messages` plus `summary`, keeping prompt size predictable at any session length.

**Intent persistence across turns:** `intent` is written on the first turn and restored by `DynamoDBSaver` on all subsequent turns. The intent classifier uses the restored value as the anchor for follow-up queries rather than re-classifying from scratch on every turn.

---

## Document Retrieval Design (Evaluation Note Path)

Retrieval uses OpenSearch native MMR (introduced in version 3.3), executing entirely inside the evaluation note MCP Lambda.

**Query type constraint:**

OpenSearch native MMR requires a `knn` or `neural` query as the top-level query in the search request. Wrapping inside a `hybrid` or `bool` query is explicitly not supported due to limitations in the hybrid search executor. The evaluation note path therefore uses pure knn search with MMR. BM25 is not used on this path.

**Query construction:**

The Lambda embeds the user query text using Titan and constructs a `knn` query against the `findings_embedding` field. The `ext.mmr` parameter is included in the request body with a configured `diversity` value and `candidates` count. OpenSearch oversamples internally, applies MMR reranking to select results that are both relevant and diverse, and returns the final top N documents.

**Optional metadata pre-filters:**

If `device_type` is present in state, it is applied as a `filter` clause on the knn query, scoping the search to documents matching that device type. If `location` is present, it is applied as an additional filter. Metadata filters are a separate mechanism from the query type and are fully compatible with both knn queries and native MMR. If neither field is in state, the query runs unfiltered against the full index.

**Empty result handling:**

If the post-MMR result set is empty, the evaluation note node performs an early exit and writes a user-facing message to `rejection_note`. The generation node is never reached without grounded context.

---

## Observability

| Signal | Mechanism | Details |
|---|---|---|
| RAG quality metrics | RAGAS + CloudWatch | Faithfulness, AnswerRelevancy, ContextPrecision emitted to `SecurityEvalRAG` namespace, dimensioned by `device_type` |
| MCP Gateway call latency and errors | CloudWatch (AgentCore runtime metrics) | Invocation count, throttled requests, error rate, end-to-end latency per endpoint |
| Node-level retry events | CloudWatch Logs | Each retry attempt logs attempt count, node name, and error type with `thread_id` |
| Graph termination on error | CloudWatch Logs | `rejection_note` content emitted as a log event with `thread_id` and `intent` on every non-success exit |
| Token streaming | FastAPI SSE | Tokens streamed progressively to frontend via StreamWriter |
| Checkpoint writes | DynamoDB | Each node execution writes a checkpoint; DynamoDB CloudWatch metrics cover write latency and throttling |
| Empty retrieval events | CloudWatch Logs | Early exits from action item node and evaluation note node on empty results logged with `thread_id`, `intent`, and filter values used |

---

## Error Handling Summary

| Condition | Node | Handling |
|---|---|---|
| Out-of-scope query | Intent classifier | Writes `rejection_note`, graph terminates immediately |
| Device ID missing | Validation | Writes error to `rejection_note`, graph terminates |
| MCP Gateway timeout | Any retrieval node | Retry up to 3 times with exponential backoff; hard error exit after 3 failures |
| Device not found in DynamoDB | Device lookup | Writes user-facing error to `rejection_note`, graph terminates |
| No action items found | Action item | Writes user-facing message to `rejection_note`, graph terminates |
| Dictionary Lambda failure | Entity extraction | Retry up to 3 times; hard error exit after 3 failures |
| Dictionary lookup no match | Entity extraction | Field deleted from state; query proceeds with reduced or no filter |
| OpenSearch empty result | Evaluation note | Writes message advising novel case to `rejection_note`, graph terminates before LLM call |
| RAGAS evaluation failure | RAGAS evaluation | Logged and dropped silently; no effect on user-facing response |