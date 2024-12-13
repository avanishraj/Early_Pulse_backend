from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List, Optional
import groq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv
load_dotenv()

groq_client = groq.Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str
    
SYSTEM_PROMPT = """You are a medical information assistant. Your role is to:
1. Provide general medical information and health education
2. Help users understand common medical terms and conditions
3. Offer general wellness and preventive health advice
4. Guide users to seek appropriate medical care when needed

Important limitations:
1. You cannot diagnose conditions or prescribe treatments
2. You must always encourage users to consult healthcare professionals for specific medical advice
3. For emergencies, you must direct users to seek immediate medical attention
4. You should be clear about your limitations as an AI assistant
5. if there is any other question which is not related to medical or healthcare, you will not answer, just say sorry I can't answer this and nothing else

Base your responses on well-established medical knowledge and trusted health organizations."""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Test(BaseModel):
    tests: list=Field(description="suggested tests based on the patient's condition")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}")
])

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="Llama3-8b-8192",
    temperature=0,  
)

def context_classifier(user_query):
    """
    Classify medical user queries into three categories:
    - NORMAL_QUERY: Regular medical-related queries
    - CHECKUP_REQUIRED: Queries indicating potential health issues
    - OTHERS: Non-medical or unnecessary inputs

    Args:
        user_query (str): The input query to be classified

    Returns:
        str: One of 'NORMAL_QUERY', 'CHECKUP_REQUIRED', or 'OTHERS'
    """
    # Retrieve Groq API key from environment variable
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # Initialize ChatGroq with specific parameters
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="Llama3-8b-8192",
        temperature=0  
    )
    
    # Comprehensive classification prompt
    classification_prompt = f"""
    You are a medical query classifier. Your task is to carefully analyze the following user query 
    and classify it into one of three specific categories:

    1. NORMAL_QUERY: 
    - General medical information questions
    - Routine health-related inquiries
    - Non-urgent medical discussions
    - Seeking general health advice
    - Questions about symptoms, treatments, or medical conditions without immediate urgency

    2. CHECKUP_REQUIRED:
    - Description of specific health symptoms
    - Expressions of ongoing health problems
    - Queries indicating potential medical conditions
    - Mentions of persistent pain, discomfort, or health concerns
    - Statements suggesting the need for medical assessment

    3. OTHERS:
    - Completely unrelated to medical topics
    - Nonsensical or irrelevant inputs
    - Non-medical conversations
    - Technical or random queries

    USER QUERY: "{user_query}"

    Respond STRICTLY with ONE of these EXACT values: 
    NORMAL_QUERY, CHECKUP_REQUIRED, or OTHERS
    """

    try:
        # Generate classification response
        response = llm.invoke(classification_prompt).content.strip()
        
        # Validate and standardize the response
        valid_responses = ['NORMAL_QUERY', 'CHECKUP_REQUIRED', 'OTHERS']
        
        if response in valid_responses:
            return response
        else:
            return 'OTHERS'
    
    except Exception as e:
        print(f"Classification error: {e}")
        return 'OTHERS'


# def checkup_classifier(user_message, available_checkup_list):
#     create the llm function that will analyse the user message and provide the test fromt the available checkup list, output must the string item only
#     IF AVAILABLE TEST IS NOT PRESENT IN THE LIST, THEN RETURN THE KEYWORD CHECKUP_NOT_PRESENT

def checkup_classifier(user_message, available_checkup_list):
    """
    Classify the appropriate checkup test based on user message and available checkup list.

    Args:
        user_message (str): The user's medical description or query
        available_checkup_list (list): List of available medical checkup tests

    Returns:
        str: Matching checkup test or 'CHECKUP_NOT_PRESENT'
    """
    # Retrieve Groq API key from environment variable
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # Initialize ChatGroq with specific parameters
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="Llama3-8b-8192",
        temperature=0.1  
    )
    
    # Comprehensive classification prompt
    classification_prompt = f"""
    ### ROLE ###
    you are an experienced doctor

    ### TASK ###
    your task is to understand the patient's condition based on the message provided.
    and you have to give the most suitable medical test available in the CHECKUP-LIST

    ### CHECKup-LIST ###
    {available_checkup_list}

    ### USER CONDITION ###
    the patient express his condition as follows: [{user_message}]

    ### INSTRUCTION ###
    1.you have to understand the pateints condition and suggest him a test from the provided
    list. if the suitable test is not present return - TEST_NOT_AVAILABLE
    other wise return the most suitable health checkup test absed on the patient's condition
    2.your output must be a python list , it must start from a square bracket and end at square bracket
    3.No starting text or suggestion output MUST start from a square bracket [].
    4.your output must be the test available in the set-[{available_checkup_list}] only, if you're unable to find the most suitable test for patient's condition then return NOT_SUITABLE_TEST_FOUND.
    5.Do not give your suggestions, return ONLY a list
    ### OUTPUT FORMAT ###
    Strictly follow the output format 
    you have to return a list , 
    example1:
    [test_1, test_2, ...]

    example2:
    [NOT_SUITABLE_TEST_FOUND]
    
    """
    # print(classification_prompt)
    prompt1 = ChatPromptTemplate.from_messages([ 
        ("system", classification_prompt),
        ("user", "{input}"),
    ])
    parser =JsonOutputParser(pydantic_object=Test)
    format_chain = (
        prompt1
        | llm
        | parser  # Ensure output is parsed as JSON
    )

    try:
        # Generate classification response
        response = format_chain.invoke({"input": classification_prompt})
        
        # Validate the response
        # print(response)
        return response
    
    except Exception as e:
        print(f"Classification error: {e}")
        return 'CHECKUP_NOT_PRESENT'

# def context_classifier(user_query):
#     classification_prompt = "Normal medical related query, isCheckupRequired"
#     create a function that uses LLM and classify the user query among one of these 3 keys: NORMAL_QUERY, CHECKUP_REQUIRED, OTHERS
#     NORMAL_QUERY: it is basically for medical related user_query
#     CHECKUP_REQUIRED: you have to check if the user is suffering from some kind of problem or explaining his condition
#     OTHERS: any non related or unecessary user input
#     create a function and also write the classification prompt based on the above description, output must be one the 3 keys, nothing else

def format_message_history(messages: List[Message]) -> List[dict]:
    """Convert message history to the format expected by Groq."""
    formatted_messages = []
    for msg in messages:
        if msg.role == "user":
            formatted_messages.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            formatted_messages.append({"role": "assistant", "content": msg.content})
    return formatted_messages


if __name__ == "__main__":
    # Test cases
    # test_queries = [
    #     "What are the symptoms of the common cold?",
    #     "I've been experiencing severe chest pain for three days",
    #     "Tell me a joke about bananas",
    #     "How can I improve my cardiovascular health?",
    #     "My head hurts constantly and I feel dizzy"
    # ]

    # for query in test_queries:
    #     result = context_classifier(query)
    #     print(f"Query: '{query}'\nClassification: {result}\n")

    checkup_list = [
        'Complete Blood Count', 
        'Lipid Profile', 
        'Thyroid Function Test', 
        'Diabetes Panel', 
        'Liver Function Test'
    ]

    # Test cases
    test_queries = [
        "I'm feeling tired and weak",
        "My cholesterol levels might be high",
    ]

    for query in test_queries:
        result = checkup_classifier(query, checkup_list)
        print(f"Query: '{query}'\nRecommended Test: {result}\n")