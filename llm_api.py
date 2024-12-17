from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

huggingface_key = os.environ.get("huggingfacehub_api_key")
groq_key = os.environ.get("groq_api_key")


def huggingface_llama3_2():
    llama3_2 = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-3B-Instruct", task="text-generation",
                              temperature=0.000001, max_new_tokens=500, huggingfacehub_api_token=huggingface_key)

    return llama3_2


def groq_llama3_1_70b():
    llama3_1_70b=ChatGroq(temperature=0.01 ,model="llama-3.3-70b-specdec",groq_api_key=groq_key)

    return llama3_1_70b

def huggingface_llama3(temprature, output_tokens):
    llama3 = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3-8B-Instruct", task="text-generation",
                              temperature=temprature, max_new_tokens=output_tokens, huggingfacehub_api_token=huggingface_key)

    return llama3