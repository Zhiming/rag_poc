from pydantic import ValidationError

from graph.action_item.prompts import (
    EXTRACT_ACTION_ITEMS,
    REJECTION_NOTE_SUFFIX,
    VALIDATION_ERROR_SUFFIX,
)
from graph.action_item.schema import ActionItemOutput
from graph.action_item.state import ActionItemAgentState
from graph.constants import (
    FIELD_DEVICE_TYPE,
    FIELD_DEVICE_TYPE_PARAGRAPHS,
    FIELD_REJECTION_NOTE,
    FIELD_RETRY_COUNT,
    FIELD_STRUCTURED_OUTPUT,
    FIELD_VALIDATION_ERRORS,
    MAX_RETRIES,
    VALIDATION_ERROR_MSG_KEY,
    VALIDATION_FAILED_MSG,
)


def invoke_llm(state: ActionItemAgentState, llm) -> dict:
    device_type = state[FIELD_DEVICE_TYPE]
    paragraphs = state[FIELD_DEVICE_TYPE_PARAGRAPHS]

    prompt = EXTRACT_ACTION_ITEMS.format(device_type=device_type, paragraphs=paragraphs)

    if state.get(FIELD_REJECTION_NOTE):
        prompt += REJECTION_NOTE_SUFFIX.format(rejection_note=state[FIELD_REJECTION_NOTE])

    if state.get(FIELD_VALIDATION_ERRORS):
        errors = "\n".join(state[FIELD_VALIDATION_ERRORS])
        prompt += VALIDATION_ERROR_SUFFIX.format(errors=errors)

    structured_llm = llm.with_structured_output(ActionItemOutput)
    response = structured_llm.invoke(prompt)

    return {FIELD_STRUCTURED_OUTPUT: response.model_dump(), FIELD_VALIDATION_ERRORS: None}


def validate_schema(state: ActionItemAgentState) -> dict:
    try:
        ActionItemOutput(**state[FIELD_STRUCTURED_OUTPUT])
        return {FIELD_VALIDATION_ERRORS: None}
    except ValidationError as e:
        errors = [err[VALIDATION_ERROR_MSG_KEY] for err in e.errors()]
        retry_count = state[FIELD_RETRY_COUNT] + 1

        if retry_count >= MAX_RETRIES:
            raise ValueError(
                VALIDATION_FAILED_MSG.format(max_retries=MAX_RETRIES, errors=errors)
            )

        return {FIELD_VALIDATION_ERRORS: errors, FIELD_RETRY_COUNT: retry_count}
