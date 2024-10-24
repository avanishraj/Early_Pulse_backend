import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

def parse_report(text_data, language):
    system_prompt = (
        f"You are a tool tasked with formatting medical report text extracted from a PDF. "
        "The text contains critical data, including blood indices and other health parameters, but it needs to be cleaned up for proper structure and readability. "
        "Your job is to remove unnecessary spaces, align values, and ensure proper medical report formatting. "
        "Preserve all critical data, values, units, and reference ranges, and ensure that the final document is well-structured and easy to read. "
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    formatted_text = chain.invoke(input=text_data)
    return formatted_text



def parse_report_2(text_data, language):
    # Define the prompts
    system_prompt_1 = (
        "You are an experienced doctor. Based on the patient's medical report, use AI tools and traditional methods to provide a clear, accurate assessment. "
        "Consider the patient's age, lifestyle, medical history, and specific details like injuries, allergies, and chronic conditions. "
        "Check if the medical test values fall within normal ranges and adhere to medical standards. "
        "Return the output as a readable summary, clearly listing key findings, values, and recommendations. "
        "Ensure the text is easy to understand and well-structured, like the example below:\n\n"
        "Haemoglobin: 14.2 g/dL - Normal range (13.5-17.5 g/dL for men, 12.0-15.5 g/dL for women)\n"
        "Calcium: 8.9 mg/dL - Normal range (8.5-10.5 mg/dL)\n"
        "Vitamin D: 22 ng/mL - Deficiency (<20 ng/mL), Insufficiency (20-29 ng/mL), Optimal (30-100 ng/mL)\n"
        "Cholesterol: 210 mg/dL - Borderline high (200-239 mg/dL)\n"
        "Blood Pressure: 135/90 mmHg - Stage 1 Hypertension (130-139/80-89 mmHg)\n\n"
        "Ensure clarity and readability in the response, without additional explanations."
    )

    # Create the prompt templates
    prompt1 = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_1),
            ("human", "{input}"),
        ]
    )

    prompt2 = ChatPromptTemplate.from_template(
        "This is the data: {data}. "
        "Please translate the entire document into {language}. "
        "Ensure that all information is accurately conveyed, maintaining the original meaning and context. "
        "Pay special attention to any technical or medical terminology to ensure precision in translation."
    )

    # Create the output parser
    output_parser = StrOutputParser()
    
    # Chain to process the analysis
    formatting_chain = prompt1 | llm | output_parser
    formatted_text = formatting_chain.invoke(input=text_data)
    
    # Clean up and refine the output if needed
    formatted_text = formatted_text.strip()  # Remove leading/trailing spaces
    
    # Translation chain
    translation_chain = prompt2 | llm | output_parser
    translated_text = translation_chain.invoke({"data": formatted_text, "language": language})
    
    # Remove any unwanted parts from the translated text
    translated_text = translated_text.strip()  # Clean up leading/trailing spaces
    
    # Return the clean, readable translation
    return translated_text