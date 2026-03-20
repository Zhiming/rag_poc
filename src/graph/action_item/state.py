from typing import Dict, List, Optional
from typing_extensions import TypedDict


class ActionItemAgentState(TypedDict):
    device_type_paragraphs: Dict[str, str]
    device_type: Optional[str]
    structured_output: Optional[dict]
    validation_errors: Optional[List[str]]
    rejection_note: Optional[str]
    retry_count: int
