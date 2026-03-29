import json
from unittest.mock import MagicMock, call, patch

import pytest
from elasticsearch import NotFoundError

from lambdas.config import EVALUATION_NOTES_INDEX_MAPPINGS
from lambdas.constants import (
    ES_FIELD_FINDINGS_TEXT,
    ES_FIELD_REMEDIATION_TEXT,
    ES_INDEX_NAME_ENV_KEY,
    ES_PASSWORD_ENV_KEY,
    ES_URL_ENV_KEY,
    ES_USERNAME_ENV_KEY,
    ES_EMBEDDED_OUTPUT_DIR_ENV_KEY,
)
from lambdas.index_evaluation_notes import handler

FAKE_ES_URL = "https://127.0.0.1:9200"
FAKE_INDEX = "evaluation_notes"

SAMPLE_NOTE = {
    ES_FIELD_FINDINGS_TEXT: "camera fault",
    ES_FIELD_REMEDIATION_TEXT: "camera replaced",
    "findings_embedding": [0.1] * 1536,
    "remediation_embedding": [0.2] * 1536,
}


def set_es_env(monkeypatch):
    monkeypatch.setenv(ES_URL_ENV_KEY, FAKE_ES_URL)
    monkeypatch.setenv(ES_USERNAME_ENV_KEY, "elastic")
    monkeypatch.setenv(ES_PASSWORD_ENV_KEY, "password")
    monkeypatch.setenv(ES_INDEX_NAME_ENV_KEY, FAKE_INDEX)


@patch("lambdas.index_evaluation_notes.Elasticsearch")
def test_drops_existing_index(mock_es_class, tmp_path, monkeypatch):
    set_es_env(monkeypatch)
    monkeypatch.setenv(ES_EMBEDDED_OUTPUT_DIR_ENV_KEY, str(tmp_path))

    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    handler({}, None)

    mock_client.indices.delete.assert_called_once_with(index=FAKE_INDEX)


@patch("lambdas.index_evaluation_notes.Elasticsearch")
def test_handles_not_found_when_index_does_not_exist(mock_es_class, tmp_path, monkeypatch):
    set_es_env(monkeypatch)
    monkeypatch.setenv(ES_EMBEDDED_OUTPUT_DIR_ENV_KEY, str(tmp_path))

    mock_client = MagicMock()
    mock_client.indices.delete.side_effect = NotFoundError(404, "index_not_found_exception", "index not found")
    mock_es_class.return_value = mock_client

    # should not raise
    handler({}, None)


@patch("lambdas.index_evaluation_notes.Elasticsearch")
def test_creates_index_with_correct_mappings(mock_es_class, tmp_path, monkeypatch):
    set_es_env(monkeypatch)
    monkeypatch.setenv(ES_EMBEDDED_OUTPUT_DIR_ENV_KEY, str(tmp_path))

    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    handler({}, None)

    mock_client.indices.create.assert_called_once_with(
        index=FAKE_INDEX,
        body=EVALUATION_NOTES_INDEX_MAPPINGS,
    )


@patch("lambdas.index_evaluation_notes.Elasticsearch")
def test_indexes_all_notes_from_all_files(mock_es_class, tmp_path, monkeypatch):
    set_es_env(monkeypatch)
    monkeypatch.setenv(ES_EMBEDDED_OUTPUT_DIR_ENV_KEY, str(tmp_path))

    (tmp_path / "file_a.json").write_text(json.dumps([SAMPLE_NOTE, SAMPLE_NOTE]))
    (tmp_path / "file_b.json").write_text(json.dumps([SAMPLE_NOTE]))

    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    handler({}, None)

    assert mock_client.index.call_count == 3
    for c in mock_client.index.call_args_list:
        assert c.kwargs["index"] == FAKE_INDEX
        assert c.kwargs["document"] == SAMPLE_NOTE


@patch("lambdas.index_evaluation_notes.Elasticsearch")
def test_returns_correct_file_and_note_counts(mock_es_class, tmp_path, monkeypatch):
    set_es_env(monkeypatch)
    monkeypatch.setenv(ES_EMBEDDED_OUTPUT_DIR_ENV_KEY, str(tmp_path))

    (tmp_path / "file_a.json").write_text(json.dumps([SAMPLE_NOTE, SAMPLE_NOTE]))
    (tmp_path / "file_b.json").write_text(json.dumps([SAMPLE_NOTE]))

    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    result = handler({}, None)

    assert result["files_indexed"] == 2
    assert result["total_notes"] == 3


@patch("lambdas.index_evaluation_notes.Elasticsearch")
def test_returns_zero_counts_for_empty_input_dir(mock_es_class, tmp_path, monkeypatch):
    set_es_env(monkeypatch)
    monkeypatch.setenv(ES_EMBEDDED_OUTPUT_DIR_ENV_KEY, str(tmp_path))

    mock_client = MagicMock()
    mock_es_class.return_value = mock_client

    result = handler({}, None)

    assert result["files_indexed"] == 0
    assert result["total_notes"] == 0
    mock_client.index.assert_not_called()
