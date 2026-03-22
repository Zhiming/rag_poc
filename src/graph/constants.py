MAX_RETRIES = 3

# ActionItemAgentState field keys
FIELD_DEVICE_TYPE_PARAGRAPHS = "device_type_paragraphs"
FIELD_DEVICE_TYPE = "device_type"
FIELD_STRUCTURED_OUTPUT = "structured_output"
FIELD_VALIDATION_ERRORS = "validation_errors"
FIELD_REJECTION_NOTE = "rejection_note"
FIELD_RETRY_COUNT = "retry_count"

# MetadataExtractionState field keys
FIELD_NORMALIZED_TEXT = "normalized_text"
FIELD_METADATA = "metadata"

# Validation
VALIDATION_ERROR_MSG_KEY = "msg"
VALIDATION_FAILED_MSG = "Schema validation failed after {max_retries} attempts: {errors}"

# Node names
NODE_INVOKE_LLM = "invoke_llm"
NODE_VALIDATE_SCHEMA = "validate_schema"
