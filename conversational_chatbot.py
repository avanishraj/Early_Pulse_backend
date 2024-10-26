from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}")
])

def format_message_history(messages: List[Message]) -> List[dict]:
    """Convert message history to the format expected by Groq."""
    formatted_messages = []
    for msg in messages:
        if msg.role == "user":
            formatted_messages.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            formatted_messages.append({"role": "assistant", "content": msg.content})
    return formatted_messages