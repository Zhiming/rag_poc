import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from graph.constants import (
    FIELD_EVALUATION_NOTES,
    FIELD_NORMALIZED_TEXT,
    FIELD_RETRY_COUNT,
    FIELD_VALIDATION_ERRORS,
)
from graph.evaluation_note_extraction.graph import EvaluationNoteExtractionGraph
from constants import (
    DEFAULT_OUTPUT_DIR,
    EVALUATION_NOTE_INPUTS_FILE,
    EVALUATION_NOTE_OUTPUT_DIR_ENV_KEY,
    OUTPUT_FILE_PREFIX,
)

load_dotenv()

inputs = json.loads(Path(EVALUATION_NOTE_INPUTS_FILE).read_text())

output_dir = Path(os.getenv(EVALUATION_NOTE_OUTPUT_DIR_ENV_KEY, DEFAULT_OUTPUT_DIR))
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

graph = EvaluationNoteExtractionGraph().build_graph()

for i, entry in enumerate(inputs, start=1):
    print(f"\n--- Input {i} ---")
    result = graph.invoke({
        FIELD_NORMALIZED_TEXT: entry[FIELD_NORMALIZED_TEXT],
        FIELD_EVALUATION_NOTES: [],
        FIELD_VALIDATION_ERRORS: None,
        FIELD_RETRY_COUNT: 0,
    })

    for j, note in enumerate(result[FIELD_EVALUATION_NOTES]):
        print(f"Note {j + 1}:")
        print(json.dumps(note, indent=2))

    output_file = output_dir / f"{OUTPUT_FILE_PREFIX}_{i}.json"
    output_file.write_text(json.dumps(result[FIELD_EVALUATION_NOTES], indent=2))
    print(f"Output written to: {output_file}")
