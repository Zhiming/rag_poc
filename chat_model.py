
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse


load_dotenv()

class Chat_Model():
    def __init__(self):
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION')
        model_id = os.getenv("MODEL_ID")

        self.__llm = ChatBedrockConverse(
            model=model_id,
            provider="anthropic",
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            temperature=0
        )

    @property
    def chat_llm(self):
        return self.__llm