import os

# --- AWS ---
DYNAMODB_SERVICE_NAME = "dynamodb"

# --- Normalization ---
UNICODE_NORMALIZATION_FORM = "NFC"
TEXT_ENCODING = "utf-8"
HASH_ALGORITHM = "sha256"
ALLOWED_CHARS_PATTERN = r"[^a-z0-9 ]"
WHITESPACE_COLLAPSE_PATTERN = r" +"

# --- DynamoDB config ---
EVALUATION_NOTE_DOCUMENT_TABLE = os.environ.get("DYNAMODB_TABLE_NAME", "")
CONTENT_HASH_INDEX = "content_hash_index"

# --- DynamoDB attribute names ---
ATTR_CONTENT_HASH = "content_hash"
ATTR_STATUS = "status"

# --- Active statuses for duplicate detection ---
ACTIVE_STATUSES = {"approved", "pending_review"}

# --- Event field keys ---
FIELD_SOURCE_INPUT = "source_input"
FIELD_NORMALIZED_TEXT = "normalized_text"
FIELD_CONTENT_HASH = "content_hash"
FIELD_STABLE_DOCUMENT_ID = "stable_document_id"
FIELD_IS_DUPLICATE = "is_duplicate"
