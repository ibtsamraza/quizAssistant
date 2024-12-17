from sentence_transformers import CrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import EnsembleRetriever
import os
from re_ranker import cross_encoder
from dotenv import load_dotenv

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
os.environ["COHERE_API_KEY"] = cohere_api_key

def hybrid_search_retriever(vector_db):

    retriever_vectordb = vector_db.as_retriever(search_kwargs={"k": 3})
    # ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
    #                                 weights=[0.5, 0.5])
    
    compressor = cross_encoder(3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever_vectordb
    )

    return compression_retriever

    


