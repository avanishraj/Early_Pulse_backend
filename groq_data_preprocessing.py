import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import json


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="Llama3-8b-8192",
    temperature=0,  
)

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Report(BaseModel):
    parameter: str=Field(description="name of the medical parameter from the report")
    observed_value: str=Field(description="value mentioned in the report along with SI unit")
    normal_upper_limit: str=Field(description="normal upper value of that medical parameter with SI unit")
    normal_lower_limit: str=Field(description="normal lower value of that medical parameter with SI unit")
    explanation: str=Field(description="Explain in very short about the paramter to a normal person")



def parse_report_2(text_data: str, language="english"):
    system_prompt_1 = """
    |<ROLE>|
    You are an experienced doctor analyzing medical reports. Format the data into a clear, structured summary with normal ranges and assessments.

    |<INSTRUCTIONS>|
    You have to return a JSON object that has a list of each finding:
    - parameter
    - observed_value : "numeric_value (SI unit)"
    - normal_upper_limit  : "numeric_value (SI unit)"
    - normal_lower_limit : "numeric_value (SI unit)"
    - explanation

    Example JSON format:
    [
         {{
            "parameter": "Haemoglobin",
            "observed_value": "14.2 (g/dL)",
            "normal_upper_limit": "17.5 (g/dL)",
            "normal_lower_limit": "13.5 (g/dL)"
            "explanation":"explanation about Haemoglobin in the given language also explain the role of Haemoglobin for the body"
        }},
        {{...}}, {{...}}
    ]

    Provide only the formatted data and translated output string without any additional message, LLM_Output, text, headers, or metadata.
    At last, provide a briefer summary in very easy and short explanation.
    """

    prompt1 = ChatPromptTemplate.from_messages([ 
        ("system", system_prompt_1),
        ("user", "{input}"),
    ])
    parser=JsonOutputParser(pydantic_object=Report)
    format_chain = (
        prompt1
        | llm
        | parser  # Ensure output is parsed as JSON
    )

    # Process the text
    try:
        # Get formatted text and clean it
        formatted_text = format_chain.invoke({"input": text_data})
        return formatted_text

    except Exception as e:
        return {"error": str(e)}  # Return error message in JSON format
    
def translate(language: str, json_data: str):
    # Define the translation system prompt
    translation_prompt = """
    |<ROLE>|
    You are an experienced doctor translating medical reports. Please translate the given report to the specified language while keeping the same structure and key names.

    |<INSTRUCTIONS>|
    You must translate the report in such a way that the following keys remain the same:
    - parameter
    - observed_value : "numeric_value (SI unit)"
    - normal_upper_limit  : "numeric_value (SI unit)"
    - normal_lower_limit : "numeric_value (SI unit)"
    - explanation

    DO NOT TRANSLATE THE KEYS

    Example JSON format (translated):
    [
        {{
            "parameter": "Haemoglobin",
            "observed_value": "14.2 (g/dL)",
            "normal_upper_limit": "17.5 (g/dL)",
            "normal_lower_limit": "13.5 (g/dL)"
            "explanation":"explanation about Haemoglobin in the given language also explain the role of Haemoglobin for the body"
        }},
        {{...}}, {{...}}
    ]

    <|IMPORTANT|>
    - Do NOT include any opening or closing statements like "Here is the output..." or "Note: I corrected...".
    - Your output MUST start and end with curly brackets, with NO additional text outside of the JSON object.
    - OUTPUT must be a JSON
    """
    parser = JsonOutputParser(pydantic_object=Report)

    chat_prompt=ChatPromptTemplate.from_messages([
        ("system",translation_prompt),
        ("human",f"""
         translate this report - {json_data} into this language - {language}, 
         keep the KEY NAMES in ENGLISH , DO NOT TRANSLATE the KEY
         """)
    ])
    try:
      chain=chat_prompt|llm|parser
      response=chain.invoke({"json_data":json_data,"language":language})
      return response

    except Exception as e:
        return {"error": str(e)}  # Return error message in JSON format

def parse_and_translate(text_data,language):
    report=parse_report_2(text_data,language)
    report=json.dumps(report)
    report=report.replace("{","{{")
    report=report.replace("}","}}")
    translation=translate(language,report)
    return translation