from graph.constants import FIELD_EVALUATION_NOTES, FIELD_NORMALIZED_TEXT
from graph.evaluation_note_extraction.prompts import EXTRACT_EVALUATION_NOTES
from graph.evaluation_note_extraction.schema import EvaluationNoteList
from graph.evaluation_note_extraction.state import EvaluationNoteExtractionState


def invoke_llm(state: EvaluationNoteExtractionState, llm) -> dict:
    prompt = EXTRACT_EVALUATION_NOTES.format(normalized_text=state[FIELD_NORMALIZED_TEXT])
    structured_llm = llm.with_structured_output(EvaluationNoteList)
    response = structured_llm.invoke(prompt)
    return {FIELD_EVALUATION_NOTES: [obs.model_dump() for obs in response.observations]}
