from lambdas.normalize_and_hash import handler, normalize


# --- normalize() unit tests ---

def test_nfc_normalization():
    # é as two code points (e + combining accent) should normalize to the same as single code point é
    assert normalize("e\u0301") == normalize("\u00e9")

def test_lowercase():
    assert normalize("HELLO WORLD") == "hello world"

def test_strip_special_chars():
    assert normalize("hello, world!") == "hello world"

def test_collapse_whitespace():
    assert normalize("hello   world") == "hello world"

def test_trim():
    assert normalize("  hello  ") == "hello"


# --- handler() integration tests ---

def test_handler_returns_all_event_fields():
    event = {"source_input": "Hello World", "pipeline_type": "evaluation_note"}
    result = handler(event, None)
    assert result["pipeline_type"] == "evaluation_note"

def test_handler_adds_normalized_text():
    event = {"source_input": "Hello World"}
    result = handler(event, None)
    assert result["normalized_text"] == "hello world"

def test_handler_adds_content_hash():
    event = {"source_input": "Hello World"}
    result = handler(event, None)
    assert "content_hash" in result
    assert len(result["content_hash"]) == 64  # SHA-256 hex digest is always 64 chars

def test_handler_same_hash_regardless_of_case():
    event1 = {"source_input": "Hello World"}
    event2 = {"source_input": "HELLO WORLD"}
    assert handler(event1, None)["content_hash"] == handler(event2, None)["content_hash"]
