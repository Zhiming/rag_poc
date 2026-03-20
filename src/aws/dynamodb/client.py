import boto3

from lambdas.constants import DYNAMODB_SERVICE_NAME

dynamodb_client = boto3.client(DYNAMODB_SERVICE_NAME)
