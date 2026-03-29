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

# --- Output dirs ---
EVALUATION_NOTE_OUTPUT_DIR_ENV_KEY = "EVALUATION_NOTE_OUTPUT_DIR"
EVALUATION_NOTE_DEFAULT_OUTPUT_DIR = "evaluation_note_output"
ES_EMBEDDED_OUTPUT_DIR_ENV_KEY = "ES_EMBEDDED_OUTPUT_DIR"
ES_EMBEDDED_DEFAULT_OUTPUT_DIR = "evaluation_note_embedded_output"

# --- Embedding ---
EMBEDDING_MODEL_ENV_KEY = "EMBEDDING_MODEL"
BEDROCK_SERVICE_NAME = "bedrock-runtime"
EMBEDDING_VECTOR_DIMS = 1536
BEDROCK_CONTENT_TYPE = "application/json"
BEDROCK_EMBED_INPUT_KEY = "inputText"
BEDROCK_EMBED_RESPONSE_BODY_KEY = "body"
BEDROCK_EMBED_OUTPUT_KEY = "embedding"

# --- Elasticsearch config ---
ES_URL_ENV_KEY = "ES_URL"
ES_USERNAME_ENV_KEY = "ES_USERNAME"
ES_PASSWORD_ENV_KEY = "ES_PASSWORD"
ES_INDEX_NAME_ENV_KEY = "ES_INDEX_NAME"
ES_DEFAULT_INDEX_NAME = "evaluation_notes"

# --- EvaluationNote input field names (from extraction graph) ---
EVAL_NOTE_FIELD_ISSUE = "issue"
EVAL_NOTE_FIELD_REMEDIATION = "remediation"
EVAL_NOTE_FIELD_DEVICE_TYPE = "device_type"
EVAL_NOTE_FIELD_MANUFACTURER = "manufacturer"
EVAL_NOTE_FIELD_DEVICE_ID = "device_id"
EVAL_NOTE_FIELD_FACILITY_ID = "facility_id"
EVAL_NOTE_FIELD_LOCATION = "location"
EVAL_NOTE_FIELD_EVALUATION_DATE = "evaluation_date"

# --- Elasticsearch index mapping ---
ES_MAPPING_KEY_MAPPINGS = "mappings"
ES_MAPPING_KEY_PROPERTIES = "properties"
ES_MAPPING_KEY_TYPE = "type"
ES_MAPPING_KEY_DIMS = "dims"
ES_MAPPING_KEY_INDEX = "index"
ES_MAPPING_KEY_SIMILARITY = "similarity"
ES_MAPPING_TYPE_TEXT = "text"
ES_MAPPING_TYPE_KEYWORD = "keyword"
ES_MAPPING_TYPE_DATE = "date"
ES_MAPPING_TYPE_DENSE_VECTOR = "dense_vector"
ES_MAPPING_SIMILARITY_COSINE = "cosine"

# --- Elasticsearch document field names ---
ES_FIELD_FINDINGS_TEXT = "findings_text"
ES_FIELD_REMEDIATION_TEXT = "remediation_text"
ES_FIELD_FINDINGS_EMBEDDING = "findings_embedding"
ES_FIELD_REMEDIATION_EMBEDDING = "remediation_embedding"
ES_FIELD_DEVICE_TYPE = "device_type"
ES_FIELD_MANUFACTURER = "manufacturer"
ES_FIELD_DEVICE_ID = "device_id"
ES_FIELD_FACILITY_ID = "facility_id"
ES_FIELD_LOCATION = "location"
ES_FIELD_EVALUATION_DATE = "evaluation_date"
