
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse


load_dotenv()

class Chat_Model():
    def __init__(self):
        aws_region = os.getenv('AWS_REGION')
        model_id = os.getenv("MODEL_ID")
        provider = os.getenv("MODEL_PROVIDER")

        self.__llm = ChatBedrockConverse(
            model=model_id,
            provider=provider ,
            region_name=aws_region,
            temperature=0
        )

    @property
    def chat_llm(self):
        return self.__llm