import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from graph.evaluation_note_extraction.graph import EvaluationNoteExtractionGraph
from constants import (
    DEFAULT_OUTPUT_DIR,
    EVALUATION_NOTE_OUTPUT_DIR_ENV_KEY,
    OUTPUT_FILE_PREFIX,
    OUTPUT_TIMESTAMP_FORMAT,
)

load_dotenv()

result = EvaluationNoteExtractionGraph().build_graph().invoke({
    "normalized_text": (
        "security evaluation conducted on 20240918 at data center dc ams 02 "

        # camera: issue + remediation -> expect entry
        "during routine inspection of the eastern server hall a hikvision dome camera "
        "with asset tag cam 4471 mounted in cage 12b was found running firmware version 5 6 2 "
        "with default admin credentials still active "
        "credentials were immediately reset and firmware was updated to version 5 8 0 "
        "motion detection alerts were re-enabled and verified "

        # security door: issue + remediation -> expect entry
        "the security door manufactured by assa abloy with id door 09 at the cage entrance "
        "showed signs of forced entry on the door frame and the locking mechanism was damaged "
        "the door was taken out of service and replaced with a temporary reinforced barrier "
        "a replacement door unit has been ordered and is scheduled for installation within 48 hours "

        # water leak sensor: issue, no remediation -> expect NO entry
        "water leak sensor sn wls 883b near the cooling unit in the mechanical room "
        "had a disconnected probe and was not reporting to the building management system "
        "the fault has been logged and is pending assignment to the facilities team "

        # personnel incident: issue + remediation -> expect entry (no device fields)
        "a contractor was found in a restricted colocation suite without an escort "
        "in violation of the visitor access policy "
        "the contractor was immediately removed from the suite and their access badge was suspended "
        "the incident was reported to the security manager and an access review was initiated "

        # environmental sensor: issue, no remediation -> expect NO entry
        "the temperature sensor in power room b was reading 12 degrees above the expected baseline "
        "the cause has not yet been identified and no corrective action has been taken"
    ),
    "evaluation_notes": [],
    "validation_errors": None,
    "retry_count": 0,
})

for i, note in enumerate(result["evaluation_notes"]):
    print(f"Note {i + 1}:")
    print(json.dumps(note, indent=2))

output_dir = Path(os.getenv(EVALUATION_NOTE_OUTPUT_DIR_ENV_KEY, DEFAULT_OUTPUT_DIR))
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime(OUTPUT_TIMESTAMP_FORMAT)
output_file = output_dir / f"{OUTPUT_FILE_PREFIX}_{timestamp}.json"

with open(output_file, "w") as f:
    json.dump(result["evaluation_notes"], f, indent=2)

print(f"\nOutput written to: {output_file}")
