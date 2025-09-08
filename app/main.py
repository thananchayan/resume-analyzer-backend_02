# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# from app.utils import extract_text_from_pdf,extract_email, extract_phone, extract_name,get_location_info,get_recommended_skills,calculate_resume_score
# import os
# import shutil
# import torch
# from app.db import SessionLocal, UserData,Feedback
# import datetime
# import pandas as pd
# from pydantic import BaseModel

# app = FastAPI()

# # Allow frontend access
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load Hugging Face Pipelines
# ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# # Load your fine-tuned classification model
# classification_model_path = "app/models/bert-role-classifier"
# classifier_model = AutoModelForSequenceClassification.from_pretrained(classification_model_path)
# classifier_tokenizer = AutoTokenizer.from_pretrained(classification_model_path)

# # Dummy label map (use yours)
# label_map = {
#     0: "Data Scientist",
#     1: "ML Engineer",
#     2: "Web Developer",
#     3: "UI/UX Designer",
#     4: "Backend Developer",
#     5: "DevOps Engineer"
# }

# @app.post("/analyze")
# async def analyze_resume(file: UploadFile = File(...)):
#     # Save file temporarily
#     save_path = f"temp_{file.filename}"
#     with open(save_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # Extract text from PDF
#     resume_text = extract_text_from_pdf(save_path)
#     os.remove(save_path)  # clean up

#     # Run NER
#     name = extract_name(resume_text)
#     email = extract_email(resume_text)
#     phone = extract_phone(resume_text)

#     # Run summarization
#     summary = summarizer(resume_text[:1024], max_length=120, min_length=30, do_sample=False)[0]['summary_text']

#     # Run job role classification
#     inputs = classifier_tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True)
#     print("input text: ",inputs)
#     outputs = classifier_model(**inputs)
#     print("output text: ",outputs)
#     pred = torch.argmax(outputs.logits, dim=1).item()
#     print("pred text: ",pred)
#     job_role = label_map[pred]

#     # Get location data from IP
#     city, state, country = get_location_info()

#     score = calculate_resume_score(resume_text, summary, get_recommended_skills(job_role), name, email, phone)


#     # Save to DB
#     db = SessionLocal()
#     user = UserData(
#     name=name or "Not detected",
#     email=email or "Not detected",
#     phone=phone or "Not detected",
#     job_role=job_role,
#     resume_score=score, 
#     summary=summary,
#     recommended_skills=", ".join(get_recommended_skills(job_role)),
#     city=city,
#     state=state,
#     country=country,
#     timestamp=datetime.datetime.utcnow()
#     )
#     db.add(user)
#     db.commit()
#     db.close()

#     # Return response
#     return {
#         "basicInfo": {
#             "name": name or "Not detected",
#             "email": email or "Not detected",
#             "phone": phone or "Not detected",
#             "experience": "Intermediate"  # (Placeholder - add logic later)
#         },
#         "summary": summary,
#         "resume_score": score,
#         "jobRole": job_role,
#         "recommendedSkills": get_recommended_skills(job_role),
     
#     }

# class FeedbackRequest(BaseModel):
#     name: str
#     email: str
#     rating: int
#     comment: str = ""


# @app.post("/feedback")
# def submit_feedback(feedback: FeedbackRequest):
#     db = SessionLocal()
#     new_feedback = Feedback(**feedback.dict())
#     db.add(new_feedback)
#     db.commit()
#     db.close()
#     return {"message": "Feedback submitted!"}


# @app.get("/admin/data")
# def get_user_data():
#     db = SessionLocal()
#     data = db.query(UserData).all()
#     db.close()
#     return [user.__dict__ for user in data]

# @app.get("/admin/feedback")
# def get_feedback():
#     db = SessionLocal()
#     data = db.query(Feedback).all()
#     db.close()
#     return [f.__dict__ for f in data]

# @app.get("/admin/export-csv")
# def export_csv():
#     db = SessionLocal()
#     data = db.query(UserData).all()
#     db.close()
#     df = pd.DataFrame([d.__dict__ for d in data])
#     df.drop("_sa_instance_state", axis=1, inplace=True)
#     path = "user_data.csv"
#     df.to_csv(path, index=False)
#     return {"csv": path}




from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from dotenv import load_dotenv
import os
  # <-- loads .env file into os.environ

import torch, re, json, os, shutil, datetime, pandas as pd

from app.utils import (
    extract_text_from_pdf,
    extract_email, extract_phone, extract_name,
    get_location_info, calculate_resume_score,preprocess_resume_text
)
from app.db import SessionLocal, UserData, Feedback
from pydantic import BaseModel

# ========================
# FastAPI setup
# ========================
app = FastAPI()
load_dotenv() 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Load Hugging Face Model (LoRA from Hub)
# ========================
HF_REPO = "thananchayan/gemma3-resume-lora"
BASE_MODEL = "google/gemma-3-270m-it"
HF_TOKEN = os.getenv("HF_TOKEN")  # <-- set this in your environment!


tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    attn_implementation="eager",   # recommended for Gemma3
    device_map="auto",
    use_auth_token=HF_TOKEN
)
model = PeftModel.from_pretrained(base, HF_REPO, use_auth_token=HF_TOKEN)
model.eval()

# ========================
# Helpers
# ========================
def make_prompt(resume_text: str):
    return (
      "You are an expert career assistant.\n"
      "TASK: Analyze the following resume and return ONLY valid JSON.\n"
      "The JSON must include:\n"
      "- job_role (string)\n"
      "- skills (array of 5–8 strings, DO NOT repeat)\n"
      "- summary (string, 2–3 complete sentences)\n\n"
      f"RESUME:\n{resume_text}\n\n"
      "JSON:"
    )

def extract_json(text: str):
    match = re.findall(r"\{.*?\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match[0])
        except:
            s = match[0].replace("'", '"')
            s = re.sub(r",\s*}", "}", s)
            s = re.sub(r",\s*]", "]", s)
            return json.loads(s)
    return {"job_role": "Unknown", "skills": [], "summary": ""}

# ========================
# API Routes
# ========================
@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    # Save and extract text
    save_path = f"temp_{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    raw_text = extract_text_from_pdf(save_path)
    cleaned_text = preprocess_resume_text(raw_text)
    cleaned_text = cleaned_text[:4000]  # cap very long resumes

    os.remove(save_path)

    # Extract basic info
    name = extract_name(raw_text)
    email = extract_email(raw_text)
    phone = extract_phone(raw_text)

    # Run LLM for role, skills, summary
    # resume_text = resume_text[:4000]  # cap very long resumes
    prompt = make_prompt(cleaned_text)

    # Tokenize with truncation
    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,   # ✅ only output length capped
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            eos_token_id=tok.eos_token_id
        )

    out_text = tok.decode(outputs[0], skip_special_tokens=True)
    parsed = extract_json(out_text)

    job_role = parsed.get("job_role", "Unknown")
    summary = parsed.get("summary", "")
    rec_skills = parsed.get("skills", [])
    rec_skills = rec_skills[:6]

    # Get location info
    city, state, country = get_location_info()

    # Calculate score
    score = calculate_resume_score(cleaned_text, summary, rec_skills, name, email, phone)

    # Save to DB
    db = SessionLocal()
    user = UserData(
        name=name or "Not detected",
        email=email or "Not detected",
        phone=phone or "Not detected",
        job_role=job_role,
        resume_score=score,
        summary=summary,
        recommended_skills=", ".join(rec_skills),
        city=city,
        state=state,
        country=country,
        timestamp=datetime.datetime.utcnow()
    )
    db.add(user)
    db.commit()
    db.close()

    # Response
    return {
        "basicInfo": {
            "name": name or "Not detected",
            "email": email or "Not detected",
            "phone": phone or "Not detected",
        },
        "summary": summary,
        "resume_score": score,
        "jobRole": job_role,
        "recommendedSkills": rec_skills,
    }

# ========================
# Feedback + Admin
# ========================
class FeedbackRequest(BaseModel):
    name: str
    email: str
    rating: int
    comment: str = ""

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    db = SessionLocal()
    new_feedback = Feedback(**feedback.dict())
    db.add(new_feedback)
    db.commit()
    db.close()
    return {"message": "Feedback submitted!"}

@app.get("/admin/data")
def get_user_data():
    db = SessionLocal()
    data = db.query(UserData).all()
    db.close()
    return [user.__dict__ for user in data]

@app.get("/admin/feedback")
def get_feedback():
    db = SessionLocal()
    data = db.query(Feedback).all()
    db.close()
    return [f.__dict__ for f in data]

@app.get("/admin/export-csv")
def export_csv():
    db = SessionLocal()
    data = db.query(UserData).all()
    db.close()
    df = pd.DataFrame([d.__dict__ for d in data])
    df.drop("_sa_instance_state", axis=1, inplace=True)
    path = "user_data.csv"
    df.to_csv(path, index=False)
    return {"csv": path}
