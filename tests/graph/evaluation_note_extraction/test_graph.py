import pytest
from unittest.mock import MagicMock

from graph.evaluation_note_extraction.graph import EvaluationNoteExtractionGraph
from graph.evaluation_note_extraction.schema import EvaluationNote


INITIAL_STATE = {
    "normalized_text": "camera with default credentials reset at hq001",
    "evaluation_notes": [],
    "validation_errors": None,
    "retry_count": 0,
}

VALID_NOTE = EvaluationNote(issue="default credentials active", remediation="credentials reset")


def _make_note_response(notes: list) -> MagicMock:
    """Mock LLM response whose evaluation_notes yield the given EvaluationNote list."""
    response = MagicMock()
    response.evaluation_notes = notes
    return response


def _bad_response() -> MagicMock:
    """Mock LLM response that produces dicts missing required fields, triggering validate_schema to fail."""
    bad_note = MagicMock()
    bad_note.model_dump.return_value = {"device_type": "camera"}  # missing issue + remediation
    response = MagicMock()
    response.evaluation_notes = [bad_note]
    return response


def _build_graph(responses: list):
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = responses
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = structured_llm

    instance = EvaluationNoteExtractionGraph.__new__(EvaluationNoteExtractionGraph)
    instance.llm = mock_llm
    return instance.build_graph(), structured_llm


def test_graph_succeeds_on_first_attempt():
    graph, structured_llm = _build_graph([_make_note_response([VALID_NOTE])])

    result = graph.invoke(INITIAL_STATE)

    assert structured_llm.invoke.call_count == 1
    assert result["evaluation_notes"][0]["issue"] == "default credentials active"
    assert result["validation_errors"] is None


def test_graph_retries_once_on_validation_failure():
    graph, structured_llm = _build_graph([_bad_response(), _make_note_response([VALID_NOTE])])

    result = graph.invoke(INITIAL_STATE)

    assert structured_llm.invoke.call_count == 2
    assert result["evaluation_notes"][0]["issue"] == "default credentials active"
    assert result["validation_errors"] is None


def test_graph_raises_after_max_retries():
    graph, structured_llm = _build_graph([_bad_response(), _bad_response(), _bad_response()])

    with pytest.raises(Exception, match="Schema validation failed"):
        graph.invoke(INITIAL_STATE)

    assert structured_llm.invoke.call_count == 3


def test_graph_appends_validation_errors_to_retry_prompt():
    graph, structured_llm = _build_graph([_bad_response(), _make_note_response([VALID_NOTE])])

    graph.invoke(INITIAL_STATE)

    retry_prompt = structured_llm.invoke.call_args_list[1][0][0]
    assert "failed schema validation" in retry_prompt
