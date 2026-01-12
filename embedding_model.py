import os
import boto3
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

source_documents = [
    "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains across multiple steps of computation in a cyclic manner.",
    "LangGraph uses a graph structure where nodes represent functions and edges represent data flow. Each node can modify the shared state.",
    "Memory in LangGraph is handled through checkpointers, which persist state across invocations. You can use MemorySaver for in-memory storage or SqliteSaver for persistent storage.",
    "Vector stores in RAG applications store embedded representations of documents. When a user asks a question, similar documents are retrieved based on semantic similarity."
]

class Context_Retriever():
    def __init__(self):
        aws_region = os.getenv('AWS_REGION')
        embedding_model = os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v1")

        bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)

        docs = [Document(page_content=text) for text in source_documents]

        # Split documents into chunks (important for long documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        split_docs = text_splitter.split_documents(docs)

        embeddings = BedrockEmbeddings(
            client=bedrock_runtime,
            model_id=embedding_model
        )

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        self.__retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    @property
    def retriever(self):
        return self.__retriever