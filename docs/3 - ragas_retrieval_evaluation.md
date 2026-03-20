# Ragas Retrieval Evaluation: Context Precision and Recall Without a Grounded Answer

## Overview

This document covers how to use Ragas to evaluate retrieval quality (context precision and context recall) in a RAG pipeline where no ground truth answer is available. The evaluation is scoped to the retrieval module only, treating it as an isolated unit: given a query, did OpenSearch return the right documents?

The test dataset is stored in DynamoDB. Each record contains a user query and a list of expected documents in plain text. There is no generated response and no ground truth answer string involved.

---

## The Ground Truth Constraint

Ragas metrics divide into two categories based on what they require at scoring time.

**Metrics that require a reference answer (`reference` field)**

These metrics, including `LLMContextRecall` and `LLMContextPrecisionWithReference`, decompose a ground truth answer string into claims and check whether the retrieved documents support those claims. They are not applicable here because the test dataset contains expected documents, not expected answers.

**Metrics that operate on expected documents (`reference_contexts` field)**

These metrics compare retrieved document texts directly against a set of expected document texts. No generated answer and no ground truth answer string is required. This is the correct path for this use case.

The `reference` field in `SingleTurnSample` is specifically for a ground truth answer string (for example `"Paris"` in response to `"What is the capital of France?"`). It is not the field for expected documents. The correct field for expected documents is `reference_contexts`, which accepts a list of plain text strings.

---

## Metric Selection

Two metrics apply directly to this use case.

### NonLLMContextPrecisionWithReference

Measures whether the documents that OpenSearch retrieved and ranked are actually relevant, with higher-ranked documents weighted more heavily. A high score means relevant documents appear at the top of the retrieved list. A low score means relevant documents are buried or irrelevant documents appear early.

This directly tests the effect of MMR reranking: if MMR is working correctly, relevant documents should appear near the top of `retrieved_contexts`.

### NonLLMContextRecall

Measures whether all the expected documents were covered by what OpenSearch returned. A score of 1.0 means every expected document was found in the retrieved set. A low score means the retrieval pipeline missed content that should have been returned.

**Important caveat on naming:** `NonLLMContextPrecisionWithReference` is misleadingly named in the Ragas documentation. Despite the "NonLLM" prefix, there is an open issue confirming the documentation is inconsistent about whether it invokes an LLM internally. If fully deterministic, zero-cost scoring is required, the alternative is `IDBasedContextPrecision` and `IDBasedContextRecall`, which compare document IDs directly with no LLM involvement. That path requires storing document IDs in DynamoDB instead of full text.

---

## DynamoDB Test Case Structure

Each DynamoDB record uses the following structure.

| Field | Type | Description |
|---|---|---|
| `test_case_name` | String (partition key) | Unique identifier for the test case |
| `query` | String | The user query passed to the retrieval pipeline |
| `documents` | List of Strings | Expected documents in plain text |

The `documents` field maps directly to `reference_contexts` in `SingleTurnSample`. No transformation is required.

---

## SingleTurnSample Field Mapping

```python
sample = SingleTurnSample(
    user_input=query,                       # from DynamoDB record["query"]
    retrieved_contexts=retrieved_contexts,  # what OpenSearch actually returned
    reference_contexts=reference_contexts,  # from DynamoDB record["documents"]
)
```

Fields not used in this evaluation path:

- `response` is not populated (no generation step)
- `reference` is not populated (no ground truth answer string)
- `rubric` is not populated

---

## Implementation

```python
import asyncio
import boto3
from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall


# fetch a single test case from DynamoDB
def fetch_test_case(table_name: str, test_case_name: str) -> dict:
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    table = dynamodb.Table(table_name)
    response = table.get_item(Key={"test_case_name": test_case_name})
    return response["Item"]


# run both metrics against a single test case
async def evaluate_retrieval(
    query: str,
    retrieved_contexts: list[str],   # what OpenSearch actually returned
    reference_contexts: list[str],   # expected documents from DynamoDB
) -> dict:
    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=retrieved_contexts,
        reference_contexts=reference_contexts,
    )

    precision_metric = NonLLMContextPrecisionWithReference()
    recall_metric = NonLLMContextRecall()

    precision_score, recall_score = await asyncio.gather(
        precision_metric.single_turn_ascore(sample),
        recall_metric.single_turn_ascore(sample),
    )

    return {
        "context_precision": precision_score,
        "context_recall": recall_score,
    }


# wire the two pieces together against your DynamoDB record
async def run_test_case(
    table_name: str,
    test_case_name: str,
    retrieved_contexts: list[str],
) -> dict:
    record = fetch_test_case(table_name, test_case_name)

    query: str = record["query"]
    reference_contexts: list[str] = record["documents"]  # plain text list from DDB

    scores = await evaluate_retrieval(
        query=query,
        retrieved_contexts=retrieved_contexts,
        reference_contexts=reference_contexts,
    )

    print(f"Test case : {test_case_name}")
    print(f"Precision : {scores['context_precision']:.4f}")
    print(f"Recall    : {scores['context_recall']:.4f}")
    return scores
```

---

## Design Notes

**Retrieval decoupling**

`retrieved_contexts` is passed in by the caller. The evaluation function has no knowledge of OpenSearch or the retrieval pipeline. The test harness is responsible for running the actual query against OpenSearch and passing the results in. This keeps evaluation logic cleanly separated from retrieval logic.

**Parallelized scoring**

Both metric calls operate independently on the same `SingleTurnSample`. Using `asyncio.gather` runs them concurrently, which matters when running a large test suite across many test cases.

**Migration path to ID-based metrics**

If `NonLLMContextPrecisionWithReference` is confirmed to invoke an LLM and fully deterministic scoring is required, the migration is straightforward:

1. Store document IDs in the DynamoDB `documents` field alongside or instead of full text.
2. Replace `NonLLMContextPrecisionWithReference` with `IDBasedContextPrecision`.
3. Replace `NonLLMContextRecall` with `IDBasedContextRecall`.
4. Replace `reference_contexts` with `reference_context_ids` and `retrieved_contexts` with `retrieved_context_ids` in `SingleTurnSample`.

The rest of the structure is identical.

---

## What This Evaluation Does Not Cover

This test setup is scoped to retrieval quality only. The following are out of scope by design.

| Metric | Why Not Applicable |
|---|---|
| `Faithfulness` | Requires a generated response |
| `ResponseRelevancy` | Requires a generated response |
| `LLMContextRecall` | Requires a ground truth answer string |
| `FactualCorrectness` | Requires a ground truth answer string |

To evaluate generation quality (faithfulness, response relevancy), a separate test harness that runs the full pipeline end to end and captures the generated response would be required.
