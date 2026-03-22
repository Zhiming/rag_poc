from typing import Dict, List, Optional

from typing_extensions import TypedDict


class MetadataExtractionState(TypedDict):
    normalized_text: str
    metadata: Optional[List[Dict]]
