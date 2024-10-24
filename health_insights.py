from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

# Define the FastAPI app
app = FastAPI()

# Define the Pydantic models
class ResponseItem(BaseModel):
    index: int
    question: str
    response: str

class OnboardingResponses(BaseModel):
    email: str
    responses: List[ResponseItem]

# Function to generate daily routine report
def generate_daily_routine_report(onboarding_data):
    system_prompt = (
        "You are a lifestyle and wellness expert. Based on the user's answers to the onboarding questions, "
        "create a comprehensive daily routine report. Consider the user's responses, habits, lifestyle choices, "
        "and preferences. Offer advice on exercise, diet, sleep, and other aspects of daily life to improve their overall well-being. "
        "Ensure the report is personalized, practical, and easy to follow."
        "do not use any brackets or do not bold any word"
        "use name = Dr. Early Pulse"
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", f"User Data: {onboarding_data}")
        ]
    )

    # Create the output parser
    output_parser = StrOutputParser()

    # Chain to process the user data and generate the routine report
    chain = prompt | llm | output_parser

    # Use the chain to invoke the LLM with the user data
    # Correctly structure the input for invoke
    routine_report = chain.invoke(input={"data": onboarding_data})

    # Clean up and refine the output if needed
    routine_report = routine_report.strip()  # Remove leading/trailing spaces
    
    return routine_report

@app.post("/generate-daily-routine")
async def generate_routine(onboarding_data: OnboardingResponses):
    try:
        # Convert the user data to a readable string for LLM input
        responses_str = ", ".join(
            [f"Question: {item.question}, Response: {item.response}" for item in onboarding_data.responses]
        )
        user_data_str = f"Email: {onboarding_data.email}, {responses_str}"
        
        # Generate the routine report
        report = generate_daily_routine_report(user_data_str)
        
        return {
            "status": "success",
            "daily_routine_report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
