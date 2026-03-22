from graph.constants import FIELD_METADATA, FIELD_NORMALIZED_TEXT
from graph.metadata_extraction.prompts import EXTRACT_METADATA
from graph.metadata_extraction.schema import EvaluationNoteList
from graph.metadata_extraction.state import MetadataExtractionState


def invoke_llm(state: MetadataExtractionState, llm) -> dict:
    prompt = EXTRACT_METADATA.format(normalized_text=state[FIELD_NORMALIZED_TEXT])
    structured_llm = llm.with_structured_output(EvaluationNoteList)
    response = structured_llm.invoke(prompt)
    return {FIELD_METADATA: [obs.model_dump() for obs in response.observations]}
