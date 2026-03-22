from typing import Dict, List, Optional

from typing_extensions import TypedDict


class EvaluationNoteExtractionState(TypedDict):
    normalized_text: str
    evaluation_notes: Optional[List[Dict]]
