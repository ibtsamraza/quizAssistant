import uuid
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
import uuid
from pydantic import BaseModel, Field
from trustcall import create_extractor
from llm_api import groq_llama3_1_70b
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from typing import List
from pydantic import BaseModel, Field

# initialize an llm
llm = groq_llama3_1_70b()

 #Json Schema for User class
class User(BaseModel):
    """
    Save notable memories the user has shared with you for later recall.
    """
    context: str = Field(
        ...,
        description=(
            "The situation or circumstance where this memory may be relevant. "
            "Include any caveats or conditions that contextualize the memory. "
            "For example, if a user shares a preference, note if it only applies "
            "in certain situations (e.g., 'only at work'). Add any other relevant "
            "'meta' details that help fully understand when and how to use this memory."
        ),
    )


class Question(BaseModel):
    question: str = Field(description="The text of the question.")
    options: List[str] = Field(description="A list of answer choices for the question.")
    correct_answer: str = Field(description="The correct answer to the question.")

class Quiz(BaseModel):
    questions: List[Question] = Field(description="A list of questions in the quiz, where each question includes the question text, options, and the correct answer.")


# create extractor for user class
def exractor(message):

    TRUSTCALL_INSTRUCTION = """Reflect on following interaction.

    Use the provided tools to retain any necessary memories which will be used later.
    Make it consicse do not repeat memories . if two different memories are same make it one

    Use parallel tool calling to handle updates and insertions simultaneously:"""

    trustcall_extractor = create_extractor(
        llm,
        tools=[User],
        tool_choice="User",
        enable_inserts=True,
    )

    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION)] + [message]))
    memory = trustcall_extractor.invoke({"messages": updated_messages,
                                        })

    
    return memory['responses'][0].context

def json_exractor(message):

    TRUSTCALL_INSTRUCTION = """covert the message to a json format"""

    trustcall_extractor = create_extractor(
        llm,
        tools=[Quiz],
        tool_choice="Quiz",
    )
    result = trustcall_extractor.invoke(
    f"""Extract the json schema from message:
    <convo>
    {message}
    </convo>"""
    )


    
    return result['responses'][0].context