from aws.dynamodb.client import dynamodb_client
from lambdas.constants import (
    ACTIVE_STATUSES,
    ATTR_CONTENT_HASH,
    ATTR_STATUS,
    CONTENT_HASH_INDEX,
    FIELD_CONTENT_HASH,
    FIELD_IS_DUPLICATE,
    FIELD_STABLE_DOCUMENT_ID,
    EVALUATION_NOTE_DOCUMENT_TABLE,
)


def handler(event: dict, context) -> dict:
    if event.get(FIELD_STABLE_DOCUMENT_ID):
        return {**event, FIELD_IS_DUPLICATE: False}

    content_hash: str = event[FIELD_CONTENT_HASH]

    response = dynamodb_client.query(
        TableName=EVALUATION_NOTE_DOCUMENT_TABLE,
        IndexName=CONTENT_HASH_INDEX,
        KeyConditionExpression=f"{ATTR_CONTENT_HASH} = :hash",
        FilterExpression=f"{ATTR_STATUS} IN (:s1, :s2)",
        ExpressionAttributeValues={
            ":hash": {"S": content_hash},
            ":s1": {"S": list(ACTIVE_STATUSES)[0]},
            ":s2": {"S": list(ACTIVE_STATUSES)[1]},
        },
        Limit=1,
    )

    is_duplicate = response["Count"] > 0

    return {**event, FIELD_IS_DUPLICATE: is_duplicate}
