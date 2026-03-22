from typing import Dict, List

from typing_extensions import TypedDict


class EvaluationNoteExtractionState(TypedDict):
    normalized_text: str
    evaluation_notes: List[Dict]
