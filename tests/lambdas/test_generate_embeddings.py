import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from lambdas.generate_embeddings import embed, enrich_note, handler
from lambdas.constants import (
    BEDROCK_CONTENT_TYPE,
    BEDROCK_EMBED_INPUT_KEY,
    ES_FIELD_DEVICE_TYPE,
    ES_FIELD_EVALUATION_DATE,
    ES_FIELD_FACILITY_ID,
    ES_FIELD_FINDINGS_EMBEDDING,
    ES_FIELD_FINDINGS_TEXT,
    ES_FIELD_LOCATION,
    ES_FIELD_MANUFACTURER,
    ES_FIELD_REMEDIATION_EMBEDDING,
    ES_FIELD_REMEDIATION_TEXT,
    EMBEDDING_MODEL_ENV_KEY,
)

FAKE_MODEL_ID = "amazon.titan-embed-text-v1"
FAKE_EMBEDDING = [0.1] * 1536


def make_bedrock_response(embedding: list[float]) -> MagicMock:
    mock_response = MagicMock()
    mock_response.__getitem__.side_effect = lambda key: BytesIO(
        json.dumps({"embedding": embedding}).encode()
    )
    return mock_response


# --- embed() ---

def test_embed_calls_bedrock_with_correct_model_id():
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)
    embed(mock_client, FAKE_MODEL_ID, "some text")
    call_kwargs = mock_client.invoke_model.call_args.kwargs
    assert call_kwargs["modelId"] == FAKE_MODEL_ID


def test_embed_calls_bedrock_with_correct_content_type():
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)
    embed(mock_client, FAKE_MODEL_ID, "some text")
    call_kwargs = mock_client.invoke_model.call_args.kwargs
    assert call_kwargs["contentType"] == BEDROCK_CONTENT_TYPE
    assert call_kwargs["accept"] == BEDROCK_CONTENT_TYPE


def test_embed_sends_text_in_body():
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)
    embed(mock_client, FAKE_MODEL_ID, "some text")
    call_kwargs = mock_client.invoke_model.call_args.kwargs
    body = json.loads(call_kwargs["body"])
    assert body[BEDROCK_EMBED_INPUT_KEY] == "some text"


def test_embed_returns_embedding_vector():
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)
    result = embed(mock_client, FAKE_MODEL_ID, "some text")
    assert result == FAKE_EMBEDDING


# --- enrich_note() ---

def make_mock_bedrock_client() -> MagicMock:
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)
    return mock_client


def test_enrich_note_maps_issue_to_findings_text():
    note = {"issue": "camera fault", "remediation": "camera replaced"}
    result = enrich_note(make_mock_bedrock_client(), FAKE_MODEL_ID, note)
    assert result[ES_FIELD_FINDINGS_TEXT] == "camera fault"


def test_enrich_note_maps_remediation_to_remediation_text():
    note = {"issue": "camera fault", "remediation": "camera replaced"}
    result = enrich_note(make_mock_bedrock_client(), FAKE_MODEL_ID, note)
    assert result[ES_FIELD_REMEDIATION_TEXT] == "camera replaced"


def test_enrich_note_includes_findings_embedding():
    note = {"issue": "camera fault", "remediation": "camera replaced"}
    result = enrich_note(make_mock_bedrock_client(), FAKE_MODEL_ID, note)
    assert result[ES_FIELD_FINDINGS_EMBEDDING] == FAKE_EMBEDDING


def test_enrich_note_includes_remediation_embedding():
    note = {"issue": "camera fault", "remediation": "camera replaced"}
    result = enrich_note(make_mock_bedrock_client(), FAKE_MODEL_ID, note)
    assert result[ES_FIELD_REMEDIATION_EMBEDDING] == FAKE_EMBEDDING


def test_enrich_note_passes_through_optional_fields():
    note = {
        "issue": "camera fault",
        "remediation": "camera replaced",
        "device_type": "camera",
        "manufacturer": "hikvision",
        "facility_id": "dc-ams-02",
        "location": "cage 12b",
        "evaluation_date": "2024-09-18",
    }
    result = enrich_note(make_mock_bedrock_client(), FAKE_MODEL_ID, note)
    assert result[ES_FIELD_DEVICE_TYPE] == "camera"
    assert result[ES_FIELD_MANUFACTURER] == "hikvision"
    assert result[ES_FIELD_FACILITY_ID] == "dc-ams-02"
    assert result[ES_FIELD_LOCATION] == "cage 12b"
    assert result[ES_FIELD_EVALUATION_DATE] == "2024-09-18"


def test_enrich_note_omits_absent_optional_fields():
    note = {"issue": "personnel incident", "remediation": "badge suspended"}
    result = enrich_note(make_mock_bedrock_client(), FAKE_MODEL_ID, note)
    assert ES_FIELD_DEVICE_TYPE not in result
    assert ES_FIELD_MANUFACTURER not in result
    assert ES_FIELD_FACILITY_ID not in result
    assert ES_FIELD_LOCATION not in result
    assert ES_FIELD_EVALUATION_DATE not in result


# --- handler() ---

@patch("lambdas.generate_embeddings.boto3.client")
def test_handler_returns_correct_counts(mock_boto3, tmp_path, monkeypatch):
    mock_boto3.return_value.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "evaluation_notes_20260328.json").write_text(
        json.dumps([
            {"issue": "camera fault", "remediation": "replaced"},
            {"issue": "door damaged", "remediation": "barrier installed"},
        ])
    )

    output_dir = tmp_path / "output"
    monkeypatch.setenv(EMBEDDING_MODEL_ENV_KEY, FAKE_MODEL_ID)
    monkeypatch.setenv("EVALUATION_NOTE_OUTPUT_DIR", str(input_dir))
    monkeypatch.setenv("ES_EMBEDDED_OUTPUT_DIR", str(output_dir))

    result = handler({}, None)

    assert result["files_processed"] == 1
    assert result["total_notes"] == 2


@patch("lambdas.generate_embeddings.boto3.client")
def test_handler_writes_enriched_file_to_output_dir(mock_boto3, tmp_path, monkeypatch):
    mock_boto3.return_value.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "evaluation_notes_20260328.json").write_text(
        json.dumps([{"issue": "camera fault", "remediation": "replaced"}])
    )

    output_dir = tmp_path / "output"
    monkeypatch.setenv(EMBEDDING_MODEL_ENV_KEY, FAKE_MODEL_ID)
    monkeypatch.setenv("EVALUATION_NOTE_OUTPUT_DIR", str(input_dir))
    monkeypatch.setenv("ES_EMBEDDED_OUTPUT_DIR", str(output_dir))

    handler({}, None)

    output_file = output_dir / "evaluation_notes_20260328.json"
    assert output_file.exists()
    enriched = json.loads(output_file.read_text())
    assert len(enriched) == 1
    assert enriched[0][ES_FIELD_FINDINGS_TEXT] == "camera fault"


@patch("lambdas.generate_embeddings.boto3.client")
def test_handler_empty_input_dir_returns_zero_counts(mock_boto3, tmp_path, monkeypatch):
    mock_boto3.return_value.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    monkeypatch.setenv(EMBEDDING_MODEL_ENV_KEY, FAKE_MODEL_ID)
    monkeypatch.setenv("EVALUATION_NOTE_OUTPUT_DIR", str(input_dir))
    monkeypatch.setenv("ES_EMBEDDED_OUTPUT_DIR", str(output_dir))

    result = handler({}, None)

    assert result["files_processed"] == 0
    assert result["total_notes"] == 0
