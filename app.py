import streamlit as st
from embedding_models import embedding_model
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from load_multiple_db import load_retriever, load_db
from retriever import hybrid_search_retriever
from prompt_template import cot_chat_prompt, question_prompt, json_prompt,  chat_prompt, chain_of_thought_prompt
from llm_api import groq_llama3_1_70b, huggingface_llama3_2
from chain import quiz_chain, chat_chain, json_chain
import uuid
from trustcall import create_extractor
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from memory import exractor, json_exractor

# Ensure embedding model is loaded and stored in session state
if 'embedding3' not in st.session_state:
    st.session_state.embedding3 = embedding_model("cpu")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Get user id from session
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
class Question(BaseModel):
    question: str = Field(description="The text of the question.")
    options: List[str] = Field(description="A list of answer choices for the question.")
    correct_answer: str = Field(description="The correct answer to the question.")

class Quiz(BaseModel):
    questions: List[Question] = Field(description="A list of questions in the quiz, where each question includes the question text, options, and the correct answer.")
if not "quiz" in st.session_state:
    st.session_state.quiz = Quiz
# Ensure vector_store is loaded and stored in session state
if "recall_vector_store" not in st.session_state:
    recall_vector_store = InMemoryVectorStore(st.session_state.embedding3)
    st.session_state.recall_vector_store = recall_vector_store

if "doc_id" not in st.session_state:
    st.session_state.doc_id = 150

if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = []
# Initialize Chroma DB (Persistent storage for embeddings)
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = chromadb.PersistentClient(path="./multiple_chromadb")
    chroma_client = st.session_state.chroma_client
    st.session_state.chromadb = load_db(chroma_client, st.session_state.embedding3)
    st.session_state.db3 = load_retriever(st.session_state.embedding3)
    st.session_state.retriever = hybrid_search_retriever(st.session_state.db3)
# retriever = st.session_state.retriever 
# chromadb = st.session_state.chromadb

# Initialize chain (Persistent storage for embeddings)
if 'quiz_chain' not in st.session_state:
    st.session_state.question_prompt = question_prompt()
    question_prompt = st.session_state.question_prompt

    st.session_state.chain_of_thought_prompt = chain_of_thought_prompt()
    chain_of_thought_prompt = st.session_state.chain_of_thought_prompt

    st.session_state.json_prompt = json_prompt()
    json_prompt = st.session_state.json_prompt

    st.session_state.cot_chat_prompt = cot_chat_prompt()
    cot_chat_prompt = st.session_state.cot_chat_prompt

    st.session_state.chat_prompt = chat_prompt()
    chat_prompt = st.session_state.chat_prompt

    st.session_state.llm = groq_llama3_1_70b()
    llm = st.session_state.llm

    st.session_state.hf_llm= huggingface_llama3_2()
    hf_llm = st.session_state.hf_llm

    #st.session_state.chat_chain = chat_chain(st.session_state.question_prompt, st.session_state.llm, st.session_state.retriever )
    st.session_state.quiz_chain = quiz_chain(chain_of_thought_prompt, llm)
    st.session_state.json_chain = json_chain(json_prompt, llm)
    st.session_state.chat_chain = chat_chain(st.session_state.retriever, cot_chat_prompt, llm,  st.session_state.user_id, st.session_state.recall_vector_store)

quiz_chain = st.session_state.quiz_chain
json_chain = st.session_state.json_chain
chat_chain = st.session_state.chat_chain

if "documents" not in st.session_state:
    st.session_state.documents = st.session_state.chromadb._collection.get()['documents']



# App title and introduction


st.set_page_config(page_title="Interactive Learning App", layout="wide")
st.title("🧠 Interactive Learning & Chat Assistant")
st.markdown("""
Welcome to your **AI-powered assistant**!  
Here’s what you can do:
- 🤖 **Chat Interface**: Ask questions, get instant answers, and interact like OpenAI ChatGPT.
- 📚 **Quiz Generator**: Test your knowledge by taking module-based quizzes, receive feedback, and learn with explanations.
""")

# save  mesages in vector store
def save_recall_memory(memory: str):
    """Save memory to vectorstore for later semantic retrieval."""
    document = Document(
        page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": st.session_state.user_id}
    )
    st.session_state.recall_vector_store.add_documents([document])
    


#retrieve previous messages from vector store
def search_recall_memories(query: str,):
    """Search for relevant memories."""
    



    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == st.session_stateuser_id
    
    retriever = st.session_state.recall_vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
    filter=_filter_function
    )
    # documents = recall_vector_store.similarity_search(
    #     query, k=3, filter=_filter_function
    # )
    documents = retriever.invoke(query)
    return [document.page_content for document in documents]

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    mode = st.radio("Choose a mode:", ["Chat Interface", "Quiz Generator"])
    st.markdown("---")
    st.subheader("About")
    st.info("This app helps you learn interactively by combining AI-powered chat and module-based quizzes. 🧑‍💻")

if mode == "Chat Interface":
    # st.markdown("<h2 style='text-align: center;'>🤖 Chat Interface</h2>", unsafe_allow_html=True)
    # st.markdown(
    #     "<div style='text-align: center; margin-bottom: 20px;'>Ask me anything, and I'll do my best to assist you!</div>", 
    #     unsafe_allow_html=True
    # )
    
    # # Initialize session state for chat history
    # if 'chat_history' not in st.session_state:
    #     st.session_state.chat_history = []

    # # Chat input area
    # user_input = st.text_input(
    #     "Type your question below:",
    #     placeholder="e.g., What is deep learning?",
    #     label_visibility="collapsed"
    # )
    
    # # Display chat history in a chat-style format
    # chat_container = st.container()
    # with chat_container:
    #     for chat in st.session_state.chat_history:
    #         st.markdown(
    #             f"<div style='background-color: #f5f5f5; border-radius: 10px; padding: 10px; margin-bottom: 10px;'>"
    #             f"<strong style='color: #0084ff;'>User:</strong> {chat['user']}</div>",
    #             unsafe_allow_html=True
    #         )
    #         st.markdown(
    #             f"<div style='background-color: #d1f7c4; border-radius: 10px; padding: 10px; margin-bottom: 10px;'>"
    #             f"<strong style='color: #005500;'>AI:</strong> {chat['ai']}</div>",
    #             unsafe_allow_html=True
    #         )
    
    # # When user submits input
    # if user_input:
    #     # Mock response (replace with your model's API call)
    #     response = chat_chain.invoke(user_input)

    #     # Update chat history
    #     st.session_state.chat_history.append({"user": user_input, "ai": response})

    #     # Clear the input field by re-running the app
    #     st.session_state.user_input = ""  # This will clear the text_input

    # # Resetting the input field
    # if 'user_input' in st.session_state:
    #     st.text_input("Type your question below:", value=st.session_state.user_input)
 
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

    st.title("💬 Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    

    if prompt := st.chat_input():

        # if not openai_api_key:
        #     st.info("Please add your OpenAI API key to continue.")
        #     st.stop()

        #client = OpenAI(api_key=openai_api_key)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # print(st.session_state.summary)
       
        response = chat_chain.invoke({"question": prompt, "summary" : st.session_state.chat_history})
        msg = response
        #print(msg)
        if len(st.session_state.chat_history) > 10:
            summary = "/n".join(st.session_state.chat_history)
            summary_message = (
                f"These are the previous message{summary}\n\n"
                "Make informed suppositions and extrapolations based on stored"
                " previous.\n"
                "Regularly reflect on past interactions to identify patterns and"
                " preferences.\n"
                "Update your mental model of the user with each new piece of"
                " information.\n"
                "Cross-reference new information with existing memories for"
                " consistency.\n"
                "Prioritize storing emotional context and personal values"
                " alongside facts.\n"
                "Use memory to anticipate needs and tailor responses to the"
                " user's style.\n"
                "Recognize and acknowledge changes in the user's situation or"
                " perspectives over time.\n"
                "analyze whole conversation criticaly and remember key points mentioned in every message. /n"
                "based on the above mental model that you have created based on the previous messages create a smmary of it . /n"
                
            )
            messages = [HumanMessage(content=summary_message)]
            st.session_state.chat_history = [st.session_state.llm.invoke(messages).content]
            
        extractor_schema = exractor(msg)
        st.session_state.chat_history.append(extractor_schema)
        print(st.session_state.chat_history)
        save_recall_memory(extractor_schema)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
        
# Quiz Generator
elif mode == "Quiz Generator":

    st.subheader("📚 Quiz Generator")

    # Ask for the number of questions
    num_questions = st.number_input("How many questions would you like to answer?", min_value=1, step=1)

    # Initialize session state variables
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0  # Tracks the current question index
    if 'question_id' not in st.session_state:
        st.session_state.question_id = 0
    if 'total_questions' not in st.session_state:
        st.session_state.total_questions = 0  # Tracks the total questions for this session
    if 'score' not in st.session_state:
        st.session_state.score = 0  # Tracks the user's score
    if 'show_next_button' not in st.session_state:
        st.session_state.show_next_button = False  # To manage when to show the next button
    if "mcq_question" not in st.session_state:
        st.session_state.mcq_question = None
    # Reset session state if a new quiz is started
    if st.session_state.total_questions != num_questions:
        st.session_state.current_question = 0
        st.session_state.total_questions = num_questions
        st.session_state.score = 0
        st.session_state.show_next_button = False

    # Module selection
    module_number = st.number_input("Enter the module number for your quiz:", min_value=1, step=1)
    if st.session_state.current_question < st.session_state.total_questions:
        if st.session_state.question_id >4 or st.session_state.question_id == 0:
            # Generate the current question
            st.session_state.question_id = 0
            st.session_state.mcq_question  = None
            doc = "/n".join(st.session_state.documents[st.session_state.doc_id:st.session_state.doc_id+3])
            st.session_state.doc_id += 3
            question_str = quiz_chain.invoke({"document": doc, "num_of_questions": 5})
            print(question_str)
            st.session_state.mcq_question = json_chain.invoke({"text_response":question_str})

        question =  st.session_state.mcq_question[st.session_state.question_id]
        # Display the question
        st.markdown(f"**Question {st.session_state.current_question + 1}:** {question['question']}")

        # Display the options
        for idx, option in enumerate(question['options'], start=1):
            st.markdown(f"{chr(64 + idx)}. {option}")
        user_answer_key = f"answer_{st.session_state.current_question}"

        if 'clear_input_flag' not in st.session_state:
            st.session_state.clear_input_flag = False

        # If the flag is set, clear the input and reset the flag
        if st.session_state.clear_input_flag:
            st.session_state[user_answer_key] = ""
            st.session_state.clear_input_flag = False
        # Get user's answer
        user_answer = st.text_input(
            "Your Answer:", 
            key=user_answer_key,
            placeholder="Enter your answer here"
        )

        # Submit answer button
        submit = st.button("Submit Answer")
        st.session_state.quiz_history.append([question['question'], user_answer, question['correct_answer']])
        if submit and user_answer:
            # Check the user's answer
            correct_option = question['correct_answer']
            is_correct = user_answer.strip().upper() == correct_option.strip().upper()

            if is_correct:
                st.success("✅ Correct!")
                st.session_state.score += 1  # Increment score for correct answers
            else:
                st.error(f"❌ Incorrect. The correct answer was {correct_option}.")


            # Provide explanation
            st.markdown(f"**Explanation:** will implement it later")
            st.session_state.clear_input_flag = True
            # Enable the "Next Question" button
            st.session_state.show_next_button = True


    # Show the next question button
    if st.session_state.show_next_button and (st.session_state.current_question < st.session_state.total_questions or st.session_state.question_id < 4):
        next_question_clicked = st.button("Next Question")

        if next_question_clicked:
            st.session_state.current_question += 1
            st.session_state.show_next_button = False  # Hide the button for the next question
            st.session_state.question_id += 1
            st.empty()
    # When all questions are completed
    if st.session_state.current_question >= st.session_state.total_questions:
        st.markdown(f"### Quiz Completed! 🎉")
        st.markdown(f"**Your Score:** {st.session_state.score} / {st.session_state.total_questions}")
        
        # Option to restart the quiz
        if st.button("Start a new quiz"):
            st.session_state.current_question = 0
            st.session_state.total_questions = 0
            st.session_state.score = 0
            st.session_state.question_id = 0
            st.session_state.show_next_button = False
