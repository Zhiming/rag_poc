from typing import List, Optional

from pydantic import BaseModel


class EvaluationNote(BaseModel):
    issue: str
    remediation: str
    device_type: Optional[str] = None
    manufacturer: Optional[str] = None
    device_id: Optional[str] = None
    facility_id: Optional[str] = None
    location: Optional[str] = None
    evaluation_date: Optional[str] = None  # ISO date string (YYYY-MM-DD)


class EvaluationNoteList(BaseModel):
    observations: List[EvaluationNote]
