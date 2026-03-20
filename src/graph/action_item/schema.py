from typing import List

from pydantic import BaseModel


class ActionItem(BaseModel):
    title: str
    description: str
    compliance_refs: List[str] = []


class ActionItemOutput(BaseModel):
    device_type: str
    action_items: List[ActionItem]
