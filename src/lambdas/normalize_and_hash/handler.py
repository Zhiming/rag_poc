import hashlib
import re
import unicodedata

from lambdas.normalize_and_hash.constants import (
    ALLOWED_CHARS_PATTERN,
    FIELD_CONTENT_HASH,
    FIELD_NORMALIZED_TEXT,
    FIELD_SOURCE_INPUT,
    HASH_ALGORITHM,
    TEXT_ENCODING,
    UNICODE_NORMALIZATION_FORM,
    WHITESPACE_COLLAPSE_PATTERN,
)


def normalize(text: str) -> str:
    text = unicodedata.normalize(UNICODE_NORMALIZATION_FORM, text)
    text = text.lower()
    text = re.sub(ALLOWED_CHARS_PATTERN, "", text)
    text = re.sub(WHITESPACE_COLLAPSE_PATTERN, " ", text)
    text = text.strip()
    return text


def handler(event: dict, context) -> dict:
    source_input: str = event[FIELD_SOURCE_INPUT]

    normalized_text = normalize(source_input)
    content_hash = hashlib.new(HASH_ALGORITHM, normalized_text.encode(TEXT_ENCODING)).hexdigest()

    return {
        **event,
        FIELD_NORMALIZED_TEXT: normalized_text,
        FIELD_CONTENT_HASH: content_hash,
    }
