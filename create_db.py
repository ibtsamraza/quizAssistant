from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import BM25Retriever

def folder_exists(path):
    return os.path.exists(path) and os.path.isdir(path)

def chroma_db(documents, vector_embeddings_directory, embedding_model):
    if not folder_exists(vector_embeddings_directory):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        vectordb3 = Chroma.from_documents(documents=texts, embedding=embedding_model, 
                    persist_directory=vector_embeddings_directory
        )
    else:
        print("\nLoading Embeddings..\n")
        # Load exisiting vector embeddings
        collection_metadata={"hnsw:space": "cosine"}
        vectordb3 = Chroma(persist_directory=vector_embeddings_directory,
                           embedding_function=embedding_model, metadata=collection_metadata)
    
    keyword_retriever = BM25Retriever.from_documents(texts)
    keyword_retriever.k =  5
    
    return vectordb3, keyword_retriever

def faiss_db(documents, vector_embeddings_directory, embedding_model):

    if not folder_exists(vector_embeddings_directory):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        vectordb3 = Chroma.from_documents(
        documents=texts, embedding=embedding_model, persist_directory=vector_embeddings_directory
        )

    else:
        print("\nLoading Embeddings..\n")
        # Load exisiting vector embeddings
        vectordb3= FAISS.load_local(vector_embeddings_directory, embedding_model, distance_strategy=DistanceStrategy.COSINE,)
    
    keyword_retriever = BM25Retriever.from_documents(texts)  
    keyword_retriever.k =  5

    return vectordb3, keyword_retriever