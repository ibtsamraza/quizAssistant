from langchain.schema.runnable import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
class Question(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

# Define a model for the entire quiz
class Quiz(BaseModel):
    questions: List[Question]

def search_recall_memories(query: str, user_id: str, recall_vector_store: InMemoryVectorStore):
    """Search for relevant memories."""
    



    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id
    
    retriever = recall_vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
    filter=_filter_function
    )
    # documents = recall_vector_store.similarity_search(
    #     query, k=3, filter=_filter_function
    # )
    documents = retriever.invoke(query)
    print(documents)
    return [document.page_content for document in documents]
 


parser = JsonOutputParser(pydantic_object=Quiz)
def chat_chain(retriever_vectordb, prompt, llm,  user_id, recall_vector_store):

    chat_chain = (
        RunnableMap(
        {
            "history": RunnableLambda(
                lambda inputs: "\n".join(search_recall_memories(inputs["question"], user_id, recall_vector_store))  # Fetch and combine history
            ),
            "context": RunnableLambda(
                lambda inputs: "\n".join(
                    [doc.page_content for doc in retriever_vectordb.get_relevant_documents(inputs["question"])]
                )
             ), # Retrieves compressed context
            "question": RunnablePassthrough(),  # Pass-through for user question
            "summary": RunnablePassthrough(),
        }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chat_chain
# def chat_chain(compression_retriever, prompt, llm):

#     chat_chain = (
        
#         {
#             "context": compression_retriever,  # Retrieves compressed context
#             "question": RunnablePassthrough(),  # Pass-through for user question
#             "history": RunnablePassthrough(),  # Pass-through for retrieved messages
#         }

#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return chat_chain
def quiz_chain(prompt, llm):

    quiz_chain = (
       
        prompt | llm | StrOutputParser()
    )

    return quiz_chain

def json_chain(prompt, llm):

    json_chain = (
       
        prompt | llm | parser
    )

    return json_chain

# def chatbot_chain(compression_retriever, prompt, llm):

#     llm_chain = LLMChain(llm=llm, prompt=prompt)
#     chatbot_chain = RunnableMap({
#         "retrieved_docs": compression_retriever,  # Use retriever to get documents
#         "question": RunnablePassthrough(),  # Pass question directly
#     }) | (lambda x: {
#         "context": "\n\n".join([doc.page_content for doc in x["retrieved_docs"]]),
#         "question": x["question"],
#     }) | llm_chain | StrOutputParser()

#     return chatbot_chain