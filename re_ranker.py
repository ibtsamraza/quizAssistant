import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain.retrievers.document_compressors import CohereRerank
from dotenv import load_dotenv

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
os.environ["COHERE_API_KEY"] = cohere_api_key

def  cross_encoder(top_n):
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    return compressor

def rankllm_reranker(top_n):
    compressor = RankLLMRerank(top_n=top_n, model="zephyr")
    return compressor


def flash_reranker():
    compressor = FlashrankRerank()
    return compressor

def cohere_reranker(top_n):
    compressor = CohereRerank(top_n=top_n)
    return compressor


