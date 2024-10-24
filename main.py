import os
from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import PyPDF2
from io import BytesIO
from analyse_medical_history import MedicalHistoryRequest, analyze_medical_history
from groq_data_preprocessing import parse_report, parse_report_2
from health_insights import OnboardingResponses, generate_daily_routine_report

app = FastAPI()

def extract_text_from_pdf(uploaded_file: UploadFile):
    pdf_bytes = uploaded_file.file.read()
    text = ""
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text() or ""
    return text

@app.post("/upload_pdf/")
async def upload_pdf(language: str = Form(...), file_name: str = Form(...), file: UploadFile = Form(...)):
    try:
        
        extracted_text = extract_text_from_pdf(file)

        if not extracted_text:
            return JSONResponse(status_code=400, content={"message": "Failed to extract text from PDF."})

        # Parse the extracted text using the language variable
        # formatted_text = parse_report(extracted_text,language)
        formatted_text = parse_report_2(extracted_text,language)

        return JSONResponse(status_code=200, content={"message": "File processed successfully.", "LLM_output": formatted_text, "language": language})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

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

@app.post("/analyze-medical-history")
async def generate_medical_history_report(history_data: MedicalHistoryRequest):
    try:
        # Convert the medical analysis data into a readable format for the LLM
        analysis_str = "; ".join(
            [f"Report: {item.LLM_output}, Language: {item.language}, Message: {item.message}" for item in history_data.analysisList]
        )
        patient_data_str = f"Email: {history_data.email}, {analysis_str}"
        
        # Generate the medical history report
        report = analyze_medical_history(patient_data_str)
        
        return {
            "status": "success",
            "medical_history_report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

