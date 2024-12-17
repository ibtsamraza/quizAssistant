from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import BM25Retriever
from sentence_transformers import util, SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
import fitz  # PyMuPDF
import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

def embedding_model(device):

    embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    embedding3 = SentenceTransformerEmbeddings(
        model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": device}
    )

    return embedding3

def folder_exists(path):
    return os.path.exists(path) and os.path.isdir(path)


def pypdf_loader(directory_path):

    loader = DirectoryLoader(
    directory_path, glob="./*/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents
def chroma_db(documents, vector_embeddings_directory, embedding3):
    if not folder_exists(vector_embeddings_directory):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        

        vectordb3 = Chroma.from_documents(documents=texts, embedding=embedding3, 
                    persist_directory=vector_embeddings_directory
        )

document_directory_path = "./name_of_the_folder"
document = pypdf_loader(document_directory_path)
embedding_modle = embedding_model("cuda")
embeddings = chroma_db(document, "./name_of_the_folder", embedding_modle)