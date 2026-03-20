EXTRACT_ACTION_ITEMS = (
    "Extract action items for device type '{device_type}' "
    "from the following compliance text:\n\n{paragraphs}\n\n"
    "Respond with a JSON object matching this schema:\n"
    '{{"device_type": "<string>", "action_items": [{{"title": "<string>", '
    '"description": "<string>", "compliance_refs": ["<string>"]}}]}}'
)

REJECTION_NOTE_SUFFIX = "\n\nAdditional instruction from reviewer: {rejection_note}"

VALIDATION_ERROR_SUFFIX = (
    "\n\nYour previous output failed schema validation with these errors:\n{errors}"
    "\nPlease fix them."
)
