from unittest.mock import MagicMock, patch

import pytest

from lambdas.exact_duplicate_check import handler


BASE_EVENT = {
    "source_input": "some text",
    "normalized_text": "some text",
    "content_hash": "abc123",
    "pipeline_type": "evaluation_note",
}


@patch("lambdas.exact_duplicate_check.dynamodb_client")
def test_skips_check_when_stable_document_id_present(mock_ddb):
    event = {**BASE_EVENT, "stable_document_id": "EVN-2024-0047"}
    result = handler(event, None)
    mock_ddb.query.assert_not_called()
    assert result["is_duplicate"] is False


@patch("lambdas.exact_duplicate_check.dynamodb_client")
def test_returns_duplicate_true_when_match_found(mock_ddb):
    mock_ddb.query.return_value = {"Count": 1, "Items": [{"content_hash": {"S": "abc123"}}]}
    result = handler(BASE_EVENT, None)
    assert result["is_duplicate"] is True


@patch("lambdas.exact_duplicate_check.dynamodb_client")
def test_returns_duplicate_false_when_no_match(mock_ddb):
    mock_ddb.query.return_value = {"Count": 0, "Items": []}
    result = handler(BASE_EVENT, None)
    assert result["is_duplicate"] is False


@patch("lambdas.exact_duplicate_check.dynamodb_client")
def test_passes_through_all_event_fields(mock_ddb):
    mock_ddb.query.return_value = {"Count": 0, "Items": []}
    result = handler(BASE_EVENT, None)
    assert result["pipeline_type"] == "evaluation_note"
    assert result["content_hash"] == "abc123"


@patch("lambdas.exact_duplicate_check.dynamodb_client")
def test_queries_with_correct_content_hash(mock_ddb):
    mock_ddb.query.return_value = {"Count": 0, "Items": []}
    handler(BASE_EVENT, None)
    call_kwargs = mock_ddb.query.call_args.kwargs
    assert call_kwargs["ExpressionAttributeValues"][":hash"] == {"S": "abc123"}
