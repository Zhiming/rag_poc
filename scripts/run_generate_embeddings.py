import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from lambdas.constants import (
    ES_EMBEDDED_DEFAULT_OUTPUT_DIR,
    ES_EMBEDDED_OUTPUT_DIR_ENV_KEY,
)
from lambdas.generate_embeddings import handler

load_dotenv()

output_dir = Path(os.getenv(ES_EMBEDDED_OUTPUT_DIR_ENV_KEY, ES_EMBEDDED_DEFAULT_OUTPUT_DIR))
if output_dir.exists():
    shutil.rmtree(output_dir)

result = handler({}, None)
print(f"\nSummary: {result}")
