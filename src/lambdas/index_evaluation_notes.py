import json
import os
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, NotFoundError, NotFoundError

from lambdas.config import EVALUATION_NOTES_INDEX_MAPPINGS
from lambdas.constants import (
    ES_DEFAULT_INDEX_NAME,
    ES_EMBEDDED_DEFAULT_OUTPUT_DIR,
    ES_EMBEDDED_OUTPUT_DIR_ENV_KEY,
    ES_INDEX_NAME_ENV_KEY,
    ES_PASSWORD_ENV_KEY,
    ES_URL_ENV_KEY,
    ES_USERNAME_ENV_KEY,
)

load_dotenv()


def handler(event: dict, context) -> dict:
    es_url = os.environ[ES_URL_ENV_KEY]
    es_username = os.environ[ES_USERNAME_ENV_KEY]
    es_password = os.environ[ES_PASSWORD_ENV_KEY]
    index_name = os.getenv(ES_INDEX_NAME_ENV_KEY, ES_DEFAULT_INDEX_NAME)
    input_dir = Path(os.getenv(ES_EMBEDDED_OUTPUT_DIR_ENV_KEY, ES_EMBEDDED_DEFAULT_OUTPUT_DIR))

    client = Elasticsearch(
        es_url,
        basic_auth=(es_username, es_password),
        verify_certs=False,
    )

    try:
        client.indices.delete(index=index_name)
        print(f"Dropped existing index: {index_name}")
    except NotFoundError:
        print(f"No existing index found: {index_name}")

    client.indices.create(index=index_name, body=EVALUATION_NOTES_INDEX_MAPPINGS)
    print(f"Created index: {index_name}")

    input_files = sorted(input_dir.glob("*.json"))
    total_notes = 0

    for input_file in input_files:
        notes = json.loads(input_file.read_text())
        for note in notes:
            client.index(index=index_name, document=note)
        total_notes += len(notes)
        print(f"Indexed {input_file.name}: {len(notes)} note(s)")

    print(f"\nDone. {len(input_files)} file(s), {total_notes} note(s) indexed into [{index_name}]")
    return {"files_indexed": len(input_files), "total_notes": total_notes}
