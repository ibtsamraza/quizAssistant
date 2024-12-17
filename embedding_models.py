from sentence_transformers import util, SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings

def embedding_model(device):

    embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    embedding3 = SentenceTransformerEmbeddings(
        model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": device}
    )

    return embedding3