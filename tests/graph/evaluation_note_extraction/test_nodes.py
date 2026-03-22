from unittest.mock import MagicMock

from graph.evaluation_note_extraction.nodes import invoke_llm
from graph.evaluation_note_extraction.schema import EvaluationNote, EvaluationNoteList


BASE_STATE = {
    "normalized_text": "network scan on hikvision camera at facility hq001 on 20240315",
    "evaluation_notes": None,
}

CAMERA_NOTE = EvaluationNote(
    issue="default credentials active",
    remediation="credentials reset and firmware updated",
    device_type="camera",
    manufacturer="hikvision",
    device_id="cam-4471",
    facility_id="hq001",
    evaluation_date="2024-03-15",
)


def _make_llm(output: EvaluationNoteList) -> MagicMock:
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = output
    llm = MagicMock()
    llm.with_structured_output.return_value = structured_llm
    return llm


def test_invoke_llm_returns_list_of_dicts():
    llm = _make_llm(EvaluationNoteList(observations=[CAMERA_NOTE]))

    result = invoke_llm(BASE_STATE, llm)

    assert isinstance(result["evaluation_notes"], list)
    assert len(result["evaluation_notes"]) == 1
    assert result["evaluation_notes"][0]["issue"] == "default credentials active"
    assert result["evaluation_notes"][0]["remediation"] == "credentials reset and firmware updated"
    assert result["evaluation_notes"][0]["device_type"] == "camera"
    assert result["evaluation_notes"][0]["manufacturer"] == "hikvision"
    assert result["evaluation_notes"][0]["facility_id"] == "hq001"
    assert result["evaluation_notes"][0]["evaluation_date"] == "2024-03-15"


def test_invoke_llm_returns_multiple_observations():
    output = EvaluationNoteList(observations=[
        EvaluationNote(issue="default credentials active", remediation="credentials reset",
                       device_type="camera", device_id="cam-01"),
        EvaluationNote(issue="damaged locking mechanism", remediation="door replaced",
                       device_type="security_door", device_id="door-09"),
        EvaluationNote(issue="contractor in restricted area without escort",
                       remediation="badge suspended and access review initiated"),
    ])
    llm = _make_llm(output)

    result = invoke_llm(BASE_STATE, llm)

    assert len(result["evaluation_notes"]) == 3
    assert result["evaluation_notes"][0]["device_type"] == "camera"
    assert result["evaluation_notes"][1]["device_type"] == "security_door"
    assert "device_type" not in result["evaluation_notes"][2]  # non-device observation


def test_invoke_llm_with_optional_fields_absent():
    output = EvaluationNoteList(observations=[
        EvaluationNote(issue="door unresponsive", remediation="door replaced", device_type="security_door")
    ])
    llm = _make_llm(output)

    result = invoke_llm(BASE_STATE, llm)

    assert result["evaluation_notes"][0]["device_type"] == "security_door"
    assert "manufacturer" not in result["evaluation_notes"][0]
    assert "device_id" not in result["evaluation_notes"][0]


def test_invoke_llm_passes_normalized_text_in_prompt():
    llm = _make_llm(EvaluationNoteList(observations=[]))
    invoke_llm(BASE_STATE, llm)

    prompt_arg = llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert BASE_STATE["normalized_text"] in prompt_arg


def test_invoke_llm_uses_structured_output_with_correct_schema():
    llm = _make_llm(EvaluationNoteList(observations=[]))
    invoke_llm(BASE_STATE, llm)

    llm.with_structured_output.assert_called_once_with(EvaluationNoteList)


def test_invoke_llm_returns_empty_list_when_no_qualifying_observations():
    llm = _make_llm(EvaluationNoteList(observations=[]))

    result = invoke_llm(BASE_STATE, llm)

    assert result["evaluation_notes"] == []


def test_invoke_llm_excludes_none_fields_from_dicts():
    output = EvaluationNoteList(observations=[
        EvaluationNote(issue="default credentials active", remediation="credentials reset", device_type="camera")
    ])
    llm = _make_llm(output)

    result = invoke_llm(BASE_STATE, llm)

    assert "manufacturer" not in result["evaluation_notes"][0]
    assert "device_id" not in result["evaluation_notes"][0]
