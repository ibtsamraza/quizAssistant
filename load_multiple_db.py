from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from embedding_models import embedding_model
from langchain.vectorstores import Chroma
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
import os


from dotenv import load_dotenv

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")


client = chromadb.PersistentClient(path="./multiple_chromadb")
device = "cpu"
embedding3 = embedding_model(device)

vector_embeddings_directory = "./db3"

def load_retriever(embedding3):
    
    # concatenated_chroma = Chroma(embedding_function=embedding3, collection_metadata={"hnsw:space": "cosine"})
    # items = concatenated_chroma._collection.get()

    # # Iterate through each subdirectory in the folder
    # data_list = client.list_collections()
    # for db in range(0, len(data_list)):

    #     # Fetch all documents and metadata
    #     data = data_list[db].get(include=['documents', 'metadatas', 'embeddings'])
    #     # Add them to the concatenated Chroma object
    #     concatenated_chroma._collection.add(embeddings=data['embeddings'], documents=data['documents'], metadatas=data['metadatas'], ids=data['ids'])
    # return concatenated_chroma
    collection_metadata={"hnsw:space": "cosine"}
    vectordb3 = Chroma(persist_directory=vector_embeddings_directory,
                           embedding_function=embedding3, collection_metadata=collection_metadata)



    return vectordb3




def load_db(client, embedding3):
    
    concatenated_chroma = Chroma(embedding_function=embedding3, collection_metadata={"hnsw:space": "cosine"})
    items = concatenated_chroma._collection.get()

    # Iterate through each subdirectory in the folder
    data_list = client.list_collections()
    for db in range(0, len(data_list)):

        # Fetch all documents and metadata
        data = data_list[db].get(include=['documents', 'metadatas', 'embeddings'])
        # Add them to the concatenated Chroma object
        concatenated_chroma._collection.add(embeddings=data['embeddings'], documents=data['documents'], metadatas=data['metadatas'], ids=data['ids'])
    return concatenated_chroma


