import json
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from lambdas.constants import (
    BEDROCK_SERVICE_NAME,
    EMBEDDING_MODEL_ENV_KEY,
    ES_DEFAULT_INDEX_NAME,
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
    ES_QUERY_SIZE,
    ES_RESULT_KEY_HITS,
    ES_RESULT_KEY_ID,
    ES_RESULT_KEY_SCORE,
    ES_URL_ENV_KEY,
    ES_USERNAME_ENV_KEY,
    EVAL_NOTE_FIELD_DEVICE_TYPE,
    EVAL_NOTE_FIELD_ISSUE,
    EVAL_NOTE_FIELD_MANUFACTURER,
    EVALUATION_NOTE_DEFAULT_OUTPUT_DIR,
    EVALUATION_NOTE_OUTPUT_DIR_ENV_KEY,
    FIELD_DUPLICATE_NOTES,
    FIELD_MATCHED_IDS,
    FIELD_TOTAL_DUPLICATES_FOUND,
    FIELD_TOTAL_NOTES_CHECKED,
    SEMANTIC_DUP_DEFAULT_THRESHOLD,
    SEMANTIC_DUP_THRESHOLD_ENV_KEY,
)
from lambdas.generate_embeddings import embed

load_dotenv()


def build_filters(note: dict) -> list[dict]:
    device_type = note.get(EVAL_NOTE_FIELD_DEVICE_TYPE)
    manufacturer = note.get(EVAL_NOTE_FIELD_MANUFACTURER)
    if device_type and manufacturer:
        return [
            {ES_QUERY_KEY_TERM: {ES_FIELD_DEVICE_TYPE: device_type}},
            {ES_QUERY_KEY_TERM: {ES_FIELD_MANUFACTURER: manufacturer}},
        ]
    return []


def build_query(issue_text: str, embedding: list[float], filters: list[dict]) -> dict:
    knn = {
        ES_QUERY_KEY_FIELD: ES_FIELD_FINDINGS_EMBEDDING,
        ES_QUERY_KEY_QUERY_VECTOR: embedding,
        ES_QUERY_KEY_NUM_CANDIDATES: ES_QUERY_NUM_CANDIDATES_VALUE,
    }
    # Hybrid search (BM25 + kNN) with RRF requires a paid Elasticsearch license.
    # query_bool = {
    #     ES_QUERY_KEY_MUST: {ES_QUERY_KEY_MATCH: {ES_FIELD_FINDINGS_TEXT: issue_text}}
    # }
    if filters:
        knn[ES_QUERY_KEY_FILTER] = filters
        # query_bool[ES_QUERY_KEY_FILTER] = filters

    return {
        # ES_QUERY_KEY_QUERY: {ES_QUERY_KEY_BOOL: query_bool},  # RRF is a paid Elasticsearch feature
        ES_QUERY_KEY_KNN: knn,
        # ES_QUERY_KEY_RANK: {ES_QUERY_KEY_RRF: {}},            # RRF is a paid Elasticsearch feature
    }


def find_matched_ids(es_client, index_name: str, query: dict, threshold: float) -> list[str]:
    response = es_client.search(index=index_name, body=query, size=ES_QUERY_SIZE)
    hits = response[ES_RESULT_KEY_HITS][ES_RESULT_KEY_HITS]
    return [hit[ES_RESULT_KEY_ID] for hit in hits if hit[ES_RESULT_KEY_SCORE] >= threshold]


def check_note(es_client, bedrock_client, model_id: str, index_name: str, note: dict, threshold: float) -> list[str]:
    issue_text = note[EVAL_NOTE_FIELD_ISSUE]
    embedding = embed(bedrock_client, model_id, issue_text)
    filters = build_filters(note)
    query = build_query(issue_text, embedding, filters)
    return find_matched_ids(es_client, index_name, query, threshold)


def handler(event: dict, context) -> dict:
    model_id = os.environ[EMBEDDING_MODEL_ENV_KEY]
    input_dir = Path(os.getenv(EVALUATION_NOTE_OUTPUT_DIR_ENV_KEY, EVALUATION_NOTE_DEFAULT_OUTPUT_DIR))
    threshold = float(os.getenv(SEMANTIC_DUP_THRESHOLD_ENV_KEY, SEMANTIC_DUP_DEFAULT_THRESHOLD))
    index_name = os.getenv(ES_INDEX_NAME_ENV_KEY, ES_DEFAULT_INDEX_NAME)

    bedrock_client = boto3.client(BEDROCK_SERVICE_NAME)
    es_client = Elasticsearch(
        os.environ[ES_URL_ENV_KEY],
        basic_auth=(os.environ[ES_USERNAME_ENV_KEY], os.environ[ES_PASSWORD_ENV_KEY]),
        verify_certs=False,
    )

    input_files = sorted(input_dir.glob("*.json"))
    duplicate_notes = []
    total_notes_checked = 0

    for input_file in input_files:
        notes = json.loads(input_file.read_text())
        for note in notes:
            matched_ids = check_note(es_client, bedrock_client, model_id, index_name, note, threshold)
            total_notes_checked += 1
            if matched_ids:
                duplicate_notes.append({**note, FIELD_MATCHED_IDS: matched_ids})
                print(f"Duplicate found in {input_file.name}: {note[EVAL_NOTE_FIELD_ISSUE][:60]} -> {matched_ids}")

    print(f"\nDone. {total_notes_checked} note(s) checked, {len(duplicate_notes)} duplicate(s) found.")
    return {
        FIELD_DUPLICATE_NOTES: duplicate_notes,
        FIELD_TOTAL_NOTES_CHECKED: total_notes_checked,
        FIELD_TOTAL_DUPLICATES_FOUND: len(duplicate_notes),
    }
