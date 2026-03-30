import json
from io import BytesIO
from unittest.mock import MagicMock, patch

from lambdas.constants import (
    EMBEDDING_MODEL_ENV_KEY,
    ES_FIELD_DEVICE_TYPE,
    ES_FIELD_FINDINGS_EMBEDDING,
    ES_FIELD_FINDINGS_TEXT,
    ES_FIELD_MANUFACTURER,
    ES_INDEX_NAME_ENV_KEY,
    ES_PASSWORD_ENV_KEY,
    ES_QUERY_KEY_BOOL,
    ES_QUERY_KEY_FIELD,
    ES_QUERY_KEY_FILTER,
    ES_QUERY_KEY_KNN,
    ES_QUERY_KEY_MATCH,
    ES_QUERY_KEY_MUST,
    ES_QUERY_KEY_NUM_CANDIDATES,
    ES_QUERY_KEY_QUERY,
    ES_QUERY_KEY_QUERY_VECTOR,
    ES_QUERY_KEY_RANK,
    ES_QUERY_KEY_RRF,
    ES_QUERY_KEY_TERM,
    ES_QUERY_NUM_CANDIDATES_VALUE,
    ES_URL_ENV_KEY,
    ES_USERNAME_ENV_KEY,
    EVAL_NOTE_FIELD_DEVICE_TYPE,
    EVAL_NOTE_FIELD_ISSUE,
    EVAL_NOTE_FIELD_MANUFACTURER,
    EVALUATION_NOTE_OUTPUT_DIR_ENV_KEY,
    FIELD_DUPLICATE_NOTES,
    FIELD_MATCHED_IDS,
    FIELD_TOTAL_DUPLICATES_FOUND,
    FIELD_TOTAL_NOTES_CHECKED,
    SEMANTIC_DUP_THRESHOLD_ENV_KEY,
)
from lambdas.semantic_duplicate_check import (
    build_filters,
    build_query,
    check_note,
    find_matched_ids,
    handler,
)

FAKE_MODEL_ID = "amazon.titan-embed-text-v1"
FAKE_EMBEDDING = [0.1] * 1536
FAKE_ES_URL = "https://127.0.0.1:9200"
FAKE_INDEX = "evaluation_notes"
FAKE_THRESHOLD = 0.02


def make_bedrock_response(embedding: list[float]) -> MagicMock:
    mock_response = MagicMock()
    mock_response.__getitem__.side_effect = lambda key: BytesIO(
        json.dumps({"embedding": embedding}).encode()
    )
    return mock_response


def make_es_search_response(hits: list[dict]) -> dict:
    return {"hits": {"hits": hits}}


def make_hit(doc_id: str, score: float) -> dict:
    return {"_id": doc_id, "_score": score, "_source": {}}


def set_env(monkeypatch, input_dir: str):
    monkeypatch.setenv(EMBEDDING_MODEL_ENV_KEY, FAKE_MODEL_ID)
    monkeypatch.setenv(EVALUATION_NOTE_OUTPUT_DIR_ENV_KEY, input_dir)
    monkeypatch.setenv(SEMANTIC_DUP_THRESHOLD_ENV_KEY, str(FAKE_THRESHOLD))
    monkeypatch.setenv(ES_URL_ENV_KEY, FAKE_ES_URL)
    monkeypatch.setenv(ES_USERNAME_ENV_KEY, "elastic")
    monkeypatch.setenv(ES_PASSWORD_ENV_KEY, "password")
    monkeypatch.setenv(ES_INDEX_NAME_ENV_KEY, FAKE_INDEX)


# --- build_filters() ---

def test_build_filters_returns_two_term_filters_when_both_fields_present():
    note = {EVAL_NOTE_FIELD_DEVICE_TYPE: "camera", EVAL_NOTE_FIELD_MANUFACTURER: "axis"}
    filters = build_filters(note)
    assert len(filters) == 2
    assert {ES_QUERY_KEY_TERM: {ES_FIELD_DEVICE_TYPE: "camera"}} in filters
    assert {ES_QUERY_KEY_TERM: {ES_FIELD_MANUFACTURER: "axis"}} in filters


def test_build_filters_returns_empty_when_device_type_absent():
    note = {EVAL_NOTE_FIELD_MANUFACTURER: "axis"}
    assert build_filters(note) == []


def test_build_filters_returns_empty_when_manufacturer_absent():
    note = {EVAL_NOTE_FIELD_DEVICE_TYPE: "camera"}
    assert build_filters(note) == []


def test_build_filters_returns_empty_when_neither_field_present():
    assert build_filters({}) == []


# --- build_query() ---

def test_build_query_includes_match_on_findings_text():
    query = build_query("camera fault", FAKE_EMBEDDING, [])
    match = query[ES_QUERY_KEY_QUERY][ES_QUERY_KEY_BOOL][ES_QUERY_KEY_MUST][ES_QUERY_KEY_MATCH]
    assert ES_FIELD_FINDINGS_TEXT in match
    assert match[ES_FIELD_FINDINGS_TEXT] == "camera fault"


def test_build_query_includes_knn_on_findings_embedding():
    query = build_query("camera fault", FAKE_EMBEDDING, [])
    knn = query[ES_QUERY_KEY_KNN]
    assert knn[ES_QUERY_KEY_FIELD] == ES_FIELD_FINDINGS_EMBEDDING
    assert knn[ES_QUERY_KEY_QUERY_VECTOR] == FAKE_EMBEDDING
    assert knn[ES_QUERY_KEY_NUM_CANDIDATES] == ES_QUERY_NUM_CANDIDATES_VALUE


def test_build_query_includes_rrf_rank():
    query = build_query("camera fault", FAKE_EMBEDDING, [])
    assert ES_QUERY_KEY_RRF in query[ES_QUERY_KEY_RANK]


def test_build_query_applies_filters_to_both_knn_and_bool_when_filters_present():
    filters = [{ES_QUERY_KEY_TERM: {ES_FIELD_DEVICE_TYPE: "camera"}}]
    query = build_query("camera fault", FAKE_EMBEDDING, filters)
    assert query[ES_QUERY_KEY_KNN][ES_QUERY_KEY_FILTER] == filters
    assert query[ES_QUERY_KEY_QUERY][ES_QUERY_KEY_BOOL][ES_QUERY_KEY_FILTER] == filters


def test_build_query_omits_filter_key_when_no_filters():
    query = build_query("camera fault", FAKE_EMBEDDING, [])
    assert ES_QUERY_KEY_FILTER not in query[ES_QUERY_KEY_KNN]
    assert ES_QUERY_KEY_FILTER not in query[ES_QUERY_KEY_QUERY][ES_QUERY_KEY_BOOL]


# --- find_matched_ids() ---

def test_find_matched_ids_returns_ids_above_threshold():
    mock_client = MagicMock()
    mock_client.search.return_value = make_es_search_response([
        make_hit("doc-1", 0.03),
        make_hit("doc-2", 0.025),
        make_hit("doc-3", 0.01),
    ])
    result = find_matched_ids(mock_client, FAKE_INDEX, {}, threshold=0.02)
    assert result == ["doc-1", "doc-2"]


def test_find_matched_ids_returns_empty_when_no_hits_above_threshold():
    mock_client = MagicMock()
    mock_client.search.return_value = make_es_search_response([
        make_hit("doc-1", 0.01),
    ])
    result = find_matched_ids(mock_client, FAKE_INDEX, {}, threshold=0.02)
    assert result == []


def test_find_matched_ids_returns_empty_when_no_hits():
    mock_client = MagicMock()
    mock_client.search.return_value = make_es_search_response([])
    result = find_matched_ids(mock_client, FAKE_INDEX, {}, threshold=0.02)
    assert result == []


# --- check_note() ---

def test_check_note_embeds_issue_text():
    mock_bedrock = MagicMock()
    mock_bedrock.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)
    mock_es = MagicMock()
    mock_es.search.return_value = make_es_search_response([])

    note = {EVAL_NOTE_FIELD_ISSUE: "camera offline", EVAL_NOTE_FIELD_DEVICE_TYPE: "camera", EVAL_NOTE_FIELD_MANUFACTURER: "axis"}
    check_note(mock_es, mock_bedrock, FAKE_MODEL_ID, FAKE_INDEX, note, FAKE_THRESHOLD)

    call_body = json.loads(mock_bedrock.invoke_model.call_args.kwargs["body"])
    assert call_body["inputText"] == "camera offline"


def test_check_note_returns_matched_ids():
    mock_bedrock = MagicMock()
    mock_bedrock.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)
    mock_es = MagicMock()
    mock_es.search.return_value = make_es_search_response([make_hit("doc-1", 0.03)])

    note = {EVAL_NOTE_FIELD_ISSUE: "camera offline"}
    result = check_note(mock_es, mock_bedrock, FAKE_MODEL_ID, FAKE_INDEX, note, FAKE_THRESHOLD)
    assert result == ["doc-1"]


def test_check_note_returns_empty_when_no_duplicates():
    mock_bedrock = MagicMock()
    mock_bedrock.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)
    mock_es = MagicMock()
    mock_es.search.return_value = make_es_search_response([])

    note = {EVAL_NOTE_FIELD_ISSUE: "unique issue"}
    result = check_note(mock_es, mock_bedrock, FAKE_MODEL_ID, FAKE_INDEX, note, FAKE_THRESHOLD)
    assert result == []


# --- handler() ---

@patch("lambdas.semantic_duplicate_check.Elasticsearch")
@patch("lambdas.semantic_duplicate_check.boto3.client")
def test_handler_returns_duplicate_notes_with_matched_ids(mock_boto3, mock_es_class, tmp_path, monkeypatch):
    set_env(monkeypatch, str(tmp_path))

    note = {EVAL_NOTE_FIELD_ISSUE: "camera fault", EVAL_NOTE_FIELD_DEVICE_TYPE: "camera", EVAL_NOTE_FIELD_MANUFACTURER: "axis"}
    (tmp_path / "notes.json").write_text(json.dumps([note]))

    mock_boto3.return_value.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)
    mock_es_class.return_value.search.return_value = make_es_search_response([make_hit("doc-1", 0.03)])

    result = handler({}, None)

    assert result[FIELD_TOTAL_NOTES_CHECKED] == 1
    assert result[FIELD_TOTAL_DUPLICATES_FOUND] == 1
    assert len(result[FIELD_DUPLICATE_NOTES]) == 1
    assert result[FIELD_DUPLICATE_NOTES][0][FIELD_MATCHED_IDS] == ["doc-1"]


@patch("lambdas.semantic_duplicate_check.Elasticsearch")
@patch("lambdas.semantic_duplicate_check.boto3.client")
def test_handler_returns_empty_duplicate_notes_when_no_matches(mock_boto3, mock_es_class, tmp_path, monkeypatch):
    set_env(monkeypatch, str(tmp_path))

    note = {EVAL_NOTE_FIELD_ISSUE: "unique issue"}
    (tmp_path / "notes.json").write_text(json.dumps([note]))

    mock_boto3.return_value.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)
    mock_es_class.return_value.search.return_value = make_es_search_response([])

    result = handler({}, None)

    assert result[FIELD_TOTAL_NOTES_CHECKED] == 1
    assert result[FIELD_TOTAL_DUPLICATES_FOUND] == 0
    assert result[FIELD_DUPLICATE_NOTES] == []


@patch("lambdas.semantic_duplicate_check.Elasticsearch")
@patch("lambdas.semantic_duplicate_check.boto3.client")
def test_handler_checks_all_notes_across_files(mock_boto3, mock_es_class, tmp_path, monkeypatch):
    set_env(monkeypatch, str(tmp_path))

    note = {EVAL_NOTE_FIELD_ISSUE: "camera fault"}
    (tmp_path / "file_a.json").write_text(json.dumps([note, note]))
    (tmp_path / "file_b.json").write_text(json.dumps([note]))

    mock_boto3.return_value.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)
    mock_es_class.return_value.search.return_value = make_es_search_response([])

    result = handler({}, None)

    assert result[FIELD_TOTAL_NOTES_CHECKED] == 3


@patch("lambdas.semantic_duplicate_check.Elasticsearch")
@patch("lambdas.semantic_duplicate_check.boto3.client")
def test_handler_returns_zero_counts_for_empty_input_dir(mock_boto3, mock_es_class, tmp_path, monkeypatch):
    set_env(monkeypatch, str(tmp_path))

    mock_boto3.return_value.invoke_model.return_value = make_bedrock_response(FAKE_EMBEDDING)
    mock_es_class.return_value.search.return_value = make_es_search_response([])

    result = handler({}, None)

    assert result[FIELD_TOTAL_NOTES_CHECKED] == 0
    assert result[FIELD_TOTAL_DUPLICATES_FOUND] == 0
    assert result[FIELD_DUPLICATE_NOTES] == []
