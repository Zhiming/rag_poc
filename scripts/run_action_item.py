import json

from graph.action_item.graph import graph

result = graph.invoke({
    "device_type_paragraphs": {
        "4.1": "All cameras must enforce password changes on first use.",
        "4.2": "Firmware updates must be applied within 30 days of release.",
    },
    "device_type": "camera",
    "structured_output": None,
    "validation_errors": None,
    "rejection_note": None,
    "retry_count": 0,
})

print(json.dumps(result["structured_output"], indent=2))
