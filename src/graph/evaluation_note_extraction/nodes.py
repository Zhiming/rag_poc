from pydantic import ValidationError

from graph.constants import (
    FIELD_EVALUATION_NOTES,
    FIELD_NORMALIZED_TEXT,
    FIELD_RETRY_COUNT,
    FIELD_VALIDATION_ERRORS,
    MAX_RETRIES,
    VALIDATION_ERROR_MSG_KEY,
    VALIDATION_FAILED_MSG,
)
from graph.evaluation_note_extraction.prompts import EXTRACT_EVALUATION_NOTES, VALIDATION_ERROR_SUFFIX
from graph.evaluation_note_extraction.schema import EvaluationNoteList
from graph.evaluation_note_extraction.state import EvaluationNoteExtractionState


def invoke_llm(state: EvaluationNoteExtractionState, llm) -> dict:
    prompt = EXTRACT_EVALUATION_NOTES.format(normalized_text=state[FIELD_NORMALIZED_TEXT])

    if state.get(FIELD_VALIDATION_ERRORS):
        errors = "\n".join(state[FIELD_VALIDATION_ERRORS])
        prompt += VALIDATION_ERROR_SUFFIX.format(errors=errors)

    structured_llm = llm.with_structured_output(EvaluationNoteList)
    response = structured_llm.invoke(prompt)
    return {FIELD_EVALUATION_NOTES: [obs.model_dump(exclude_none=True) for obs in response.observations], FIELD_VALIDATION_ERRORS: None}


def validate_schema(state: EvaluationNoteExtractionState) -> dict:
    try:
        EvaluationNoteList(observations=state[FIELD_EVALUATION_NOTES])
        return {FIELD_VALIDATION_ERRORS: None}
    except ValidationError as e:
        errors = [err[VALIDATION_ERROR_MSG_KEY] for err in e.errors()]
        retry_count = state[FIELD_RETRY_COUNT] + 1

        if retry_count >= MAX_RETRIES:
            raise ValueError(
                VALIDATION_FAILED_MSG.format(max_retries=MAX_RETRIES, errors=errors)
            )

        return {FIELD_VALIDATION_ERRORS: errors, FIELD_RETRY_COUNT: retry_count}
