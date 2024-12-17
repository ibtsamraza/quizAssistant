from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# Define the model for each question
class Question(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

# Define a model for the entire quiz
class Quiz(BaseModel):
    questions: List[Question]

#parser = JsonOutputParser(pydantic_object=Question)
def summary_prompt():
    summary_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>  you are  my assistant who is going to summarize the text which i will provide you.
    carefully analyze the text and make sure the summary has the same context as the orignal text and summary should be like that if someone read summary it
    will have complete understanding of orignal text
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Question: {question}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    )
    return summary_prompt

def question_prompt():
    question_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an AI model that generates multiple-choice questions (MCQs) for assessments. You will receive a summary of a text, and your task is to create a set of MCQs based on this summary. Each question should have four options: one correct answer and three incorrect ones. The number of questions to generate will be specified. Please ensure that the questions are relevant and accurately reflect the content of the summary.
    return a json object

    Instructions:
    1. Generate {num_questions} multiple-choice questions based on the provided summary.
    2. Each question should be followed by four options.
    3. Clearly indicate which option is the correct answer.
    4. Make sure that the incorrect options are plausible but not correct.
    5.The following text contains multiple-choice questions (MCQs) and their answers. return a dictionary  where each question text is a key, and the options (including the correct answer) are the values in a list. Do not modify the text of the questions or options.

    Return a **valid JSON object** formatted as follows:


    # Example format:
    # Q1. [Question text]
    # A) [Option 1]
    # B) [Option 2]
    # C) [Option 3]
    # D) [Option 4]

    # Correct Answer: [Letter of the correct option]

    repeat the format for each question until the specified number of questions is generated.

    Please ensure the JSON is properly formatted, with no missing brackets or syntax errors.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Summary: {summary}
    Number of Questions: {num_questions}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["summary", "num_questions"],
    #partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return question_prompt

def table_summary_prompt():
    table_summary_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant tasked with summarizing tables and text. Give a concise summary of the table or text. Table or text chunk: {element}
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["element"],
    #partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return table_summary_prompt

def json_prompt():
    parser = JsonOutputParser(pydantic_object=Question)
    json_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>  Here is a text response I received, but it needs to be converted into a valid JSON format. If there are any formatting or structure issues in the text, please fix them and ensure that the output is a properly formatted JSON object.
    1.always output in a structured JSON format that is provided to you
    2. return a list of dictionry where each dictionary contains question, options and correct answer
    2.All the keys must be in lowercase
    3.Only use the following keys: question, options, correct_answer
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Question: {text_response}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["text_response"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return json_prompt


def memory_summary_prompt():
    memory_summary_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant designed to summarize messages efficiently and effectively, ensuring all relevant context is retained for future interactions. Your task is to condense the user's inputs into a concise summary that captures:

    User Information: Any details the user shares about themselves (e.g., skills, projects, interests, etc.).
    Ongoing Tasks/Questions: Important questions, challenges, or tasks the user is working on.
    Important Context: Key details or background information about previous interactions or projects.
    Specific Requests: Anything the user explicitly asks for or needs.

When summarizing, ensure:

    Critical details are not lost.
    Summaries are clear, concise, and logically organized.
    Information relevant to the current or potential future questions is included.

You will receive two inputs:

    Previous Summary: A summary of past conversations.
    New Message: A new message or set of messages from the user.

Your task: Merge the previous summary and the new messages into an updated, cohesive summary while removing redundant information. The final summary must retain all necessary details for the assistant to provide accurate and context-aware responses in the future.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Previous_summary: {previous_summary}
    New_message: {new_message}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    )
    return memory_summary_prompt

# def cot_chat_prompt():

#     chat_prompt = PromptTemplate(
#     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

#     You are an intelligent assistant capable of analyzing multiple inputs and synthesizing coherent answers. 
# Do not repeat or include the prompt text in your response. Only provide an answer based on your understanding of the given context.

# ### Inputs:
# 1. **Context**: Background information relevant to the question.
# 2. **Long-Term Memory (LTM)**: A collection of past knowledge and interactions stored over time. 
# 3. **Short-Term Memory (STM)**: Recent interactions or details immediately relevant to the current question. Use them if user query is relevant to previous interaction
# 4. **Question**: The user's inquiry or the problem that needs solving.

# ### Task:
# - Analyze the provided inputs carefully and provide a detailed answer which can will explain user question.
# - Do not explain the reasoning or steps.
# - Only give answer to the user and take you internall working private
# - User just want to know the answer he should not know about how you come up with the answer
# Do not include:
# - The structure or steps from the template.
# - Any explanation or reasoning steps.
# - The prompt text in your response.

# ### Inputs:
# - **Question**: {question}
# - **Context**: {context}
# - **Long-Term Memory**: {history}
# - **Short-Term Memory**: {summary}

# Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
#     input_variables=["question", "context", "history", "summary"],
# )

#     return chat_prompt

def cot_chat_prompt():
    
    chat_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an intelligent assistant capable of analyzing multiple inputs and synthesizing coherent answers. You will be provided with the following inputs: and base on the inputs you have to generate answer
    Do not add prompt text into the answer. just give an answer based on your understanding of the given context
1. **Context**: Background information relevant to the question.
2. **Long-Term Memory (LTM)**: A collection of past knowledge and interactions stored over time.
3. **Short-Term Memory (STM)**: Recent interactions or details immediately relevant to the current question.
4. **Question**: The user's inquiry or the problem that needs solving.

Your task is to:
1. **Analyze** each input carefully and extract key insights or details.
2. **Create a mental map** by connecting related information from the context, long-term memory, and short-term memory. Identify patterns, dependencies, or contradictions.
3. **Generate an accurate, logical, and user-focused answer** based on this mental map.

Follow this structure to ensure clarity:
1. **Step 1: Analyze the Inputs**
   - Summarize key information from each input.
   - Identify how the context, LTM, and STM relate to the question.

2. **Step 2: Build the Mental Map**
   - Connect the extracted details from all three memories and the context.
   - Highlight connections, gaps, or dependencies.
   - Resolve any contradictions logically.

3. **Step 3: Answer the Question**
   - Provide a clear, concise, and well-reasoned response.
   - Ensure your answer incorporates relevant details from all sources.
Analyze all the above information on your own do not show it to user
only give answer to user which is relevant to user query
Be meticulous, logical, and user-centric in your analysis. Respond only after completing the full reasoning process.

    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Long-Term Memory: {history}
    Short-Term Memory: {summary}
    
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context", "history", "summary"],
    )

    return chat_prompt
def chain_of_thought_prompt():

    chat_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an intelligent and creative quiz master. Your task is to analyze the provided context, understand its meaning, summarize the main ideas, and generate insightful and accurate quiz questions. Follow these steps carefully:

    ### Step 1: Understand the Context
    - Read the context thoroughly and understand the content.
    - Identify the key ideas, concepts, or facts that the context communicates.
    - Focus on important terms, definitions, or processes that could form the basis of quiz questions.

    ### Step 2: Summarize the Context
    - Write a concise summary of the key points in the context. Ensure the summary covers all critical aspects while remaining easy to understand.
    - Summary should only contain the key points that you thing are necessary to build good quality question

    ### Step 3: Generate Quiz Questions
    - Create quiz questions based on the context. Make sure the questions are:
    - Thought provoking and challenging.
    - Test the understanding and knowledge  of the test taker.
    - They should not be straightforward Like 'what is the the topis discussed in the context.
    - Question should be like if some profesor of that subject which has complete understanding of the topic has created it 
    - Relevant to the context.
    - Designed to assess understanding, application, or critical thinking.
    - For multiple-choice questions:
    - Provide four options , with one correct answer and three plausible distractors.
    - Do not add (A, B, C, D)
    - Clearly indicate the correct option.
    - Ensure the question targets a specific piece of information from the context.

    #### Quiz Questions:
    1. **[Question text]**
    -  [Option 1]
    -  [Option 2]
    -  [Option 3]
    -  [Option 4]
    - **Correct Answer:** [Correct Option]

    2. **[Next question]**

    ---

    ### Now process the following context and generate quiz questions:
    

    <|start_header_id|>user<|end_header_id|>
    Dcument: {document}
    num_of_questions: {num_of_questions}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["document", "num_of_questions"],
    )

    return chat_prompt

def memory_tool():
    # Define the prompt template for the agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant with advanced long-term memory"
                " capabilities. Powered by a stateless LLM, you must rely on"
                " external memory to store information between conversations."
                " Utilize the available memory tools to store and retrieve"
                " important details that will help you better attend to the user's"
                " needs and understand their context.\n\n"
                "Memory Usage Guidelines:\n"
                "1. Actively use memory tools (save_core_memory, save_recall_memory)"
                " to build a comprehensive understanding of the user.\n"
                "2. Make informed suppositions and extrapolations based on stored"
                " memories.\n"
                "3. Regularly reflect on past interactions to identify patterns and"
                " preferences.\n"
                "4. Update your mental model of the user with each new piece of"
                " information.\n"
                "5. Cross-reference new information with existing memories for"
                " consistency.\n"
                "6. Prioritize storing emotional context and personal values"
                " alongside facts.\n"
                "7. Use memory to anticipate needs and tailor responses to the"
                " user's style.\n"
                "8. Recognize and acknowledge changes in the user's situation or"
                " perspectives over time.\n"
                "9. Leverage memories to provide personalized examples and"
                " analogies.\n"
                "10. Recall past challenges or successes to inform current"
                " problem-solving.\n\n"
                "## Recall Memories\n"
                "Recall memories are contextually retrieved based on the current"
                " conversation:\n{recall_memories}\n\n"
                "## Instructions\n"
                "Engage with the user naturally, as a trusted colleague or friend."
                " There's no need to explicitly mention your memory capabilities."
                " Instead, seamlessly incorporate your understanding of the user"
                " into your responses. Be attentive to subtle cues and underlying"
                " emotions. Adapt your communication style to match the user's"
                " preferences and current emotional state. Use tools to persist"
                " information you want to retain in the next conversation. If you"
                " do call tools, all text preceding the tool call is an internal"
                " message. Respond AFTER calling the tool, once you have"
                " confirmation that the tool completed successfully.\n\n",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    return prompt


def chat_prompt():
    chat_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question and give response from the context given to you as truthfully as you can.
    Do not add anything  from you and If you don't know the answer, just say that you don't know.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Chat History: {history}
    Chat Summary: {summary} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context", "history", "summary"],
)
    
    return chat_prompt