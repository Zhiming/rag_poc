import json
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

from lambdas.constants import (
    BEDROCK_CONTENT_TYPE,
    BEDROCK_EMBED_INPUT_KEY,
    BEDROCK_EMBED_OUTPUT_KEY,
    BEDROCK_EMBED_RESPONSE_BODY_KEY,
    BEDROCK_SERVICE_NAME,
    EMBEDDING_MODEL_ENV_KEY,
    ES_EMBEDDED_DEFAULT_OUTPUT_DIR,
    ES_EMBEDDED_OUTPUT_DIR_ENV_KEY,
    ES_FIELD_DEVICE_ID,
    ES_FIELD_DEVICE_TYPE,
    ES_FIELD_EVALUATION_DATE,
    ES_FIELD_FACILITY_ID,
    ES_FIELD_FINDINGS_EMBEDDING,
    ES_FIELD_FINDINGS_TEXT,
    ES_FIELD_LOCATION,
    ES_FIELD_MANUFACTURER,
    ES_FIELD_REMEDIATION_EMBEDDING,
    ES_FIELD_REMEDIATION_TEXT,
    EVAL_NOTE_FIELD_DEVICE_ID,
    EVAL_NOTE_FIELD_DEVICE_TYPE,
    EVAL_NOTE_FIELD_EVALUATION_DATE,
    EVAL_NOTE_FIELD_FACILITY_ID,
    EVAL_NOTE_FIELD_ISSUE,
    EVAL_NOTE_FIELD_LOCATION,
    EVAL_NOTE_FIELD_MANUFACTURER,
    EVAL_NOTE_FIELD_REMEDIATION,
    EVALUATION_NOTE_DEFAULT_OUTPUT_DIR,
    EVALUATION_NOTE_OUTPUT_DIR_ENV_KEY,
)

load_dotenv()

PASS_THROUGH_FIELDS = {
    EVAL_NOTE_FIELD_DEVICE_TYPE: ES_FIELD_DEVICE_TYPE,
    EVAL_NOTE_FIELD_MANUFACTURER: ES_FIELD_MANUFACTURER,
    EVAL_NOTE_FIELD_DEVICE_ID: ES_FIELD_DEVICE_ID,
    EVAL_NOTE_FIELD_FACILITY_ID: ES_FIELD_FACILITY_ID,
    EVAL_NOTE_FIELD_LOCATION: ES_FIELD_LOCATION,
    EVAL_NOTE_FIELD_EVALUATION_DATE: ES_FIELD_EVALUATION_DATE,
}


def embed(bedrock_client, model_id: str, text: str) -> list[float]:
    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType=BEDROCK_CONTENT_TYPE,
        accept=BEDROCK_CONTENT_TYPE,
        body=json.dumps({BEDROCK_EMBED_INPUT_KEY: text}),
    )
    return json.loads(response[BEDROCK_EMBED_RESPONSE_BODY_KEY].read())[BEDROCK_EMBED_OUTPUT_KEY]


def enrich_note(bedrock_client, model_id: str, note: dict) -> dict:
    findings_text = note[EVAL_NOTE_FIELD_ISSUE]
    remediation_text = note[EVAL_NOTE_FIELD_REMEDIATION]

    enriched = {
        ES_FIELD_FINDINGS_TEXT: findings_text,
        ES_FIELD_REMEDIATION_TEXT: remediation_text,
        ES_FIELD_FINDINGS_EMBEDDING: embed(bedrock_client, model_id, findings_text),
        ES_FIELD_REMEDIATION_EMBEDDING: embed(bedrock_client, model_id, remediation_text),
    }

    for source_field, es_field in PASS_THROUGH_FIELDS.items():
        value = note.get(source_field)
        if value is not None:
            enriched[es_field] = value

    return enriched


def handler(event: dict, context) -> dict:
    model_id = os.environ[EMBEDDING_MODEL_ENV_KEY]
    input_dir = Path(os.getenv(EVALUATION_NOTE_OUTPUT_DIR_ENV_KEY, EVALUATION_NOTE_DEFAULT_OUTPUT_DIR))
    output_dir = Path(os.getenv(ES_EMBEDDED_OUTPUT_DIR_ENV_KEY, ES_EMBEDDED_DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    bedrock_client = boto3.client(BEDROCK_SERVICE_NAME)

    input_files = sorted(input_dir.glob("*.json"))
    total_notes = 0

    for input_file in input_files:
        notes = json.loads(input_file.read_text())
        enriched_notes = [enrich_note(bedrock_client, model_id, note) for note in notes]

        output_file = output_dir / input_file.name
        output_file.write_text(json.dumps(enriched_notes, indent=2))

        total_notes += len(enriched_notes)
        print(f"Processed {input_file.name}: {len(enriched_notes)} note(s) -> {output_file}")

    print(f"\nDone. {len(input_files)} file(s), {total_notes} note(s) written to {output_dir}")
    return {"files_processed": len(input_files), "total_notes": total_notes}
