# main.py
import os
import re
import tempfile
import requests
import email
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, Form, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer
from bs4 import BeautifulSoup
import fitz
import extract_msg
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorstore import embed_to_pinecone, get_relevant_context
from llm import (
    get_llm_answer,
    detect_language,
    translate_to_english,
    translate_answer
)

AUTH_KEY = os.getenv("AUTH_KEY")

app = FastAPI(
    title="RAG Compliance Assistant",
    version="1.0.0"
)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

security = HTTPBearer()

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

def verify_token(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(401, "Invalid token")
    if auth.split("Bearer ")[1] != AUTH_KEY:
        raise HTTPException(403, "Unauthorized")

def clean_text(text: str) -> str:
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def extract_pdf(data: bytes):
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "".join(p.get_text() for p in doc)

def extract_docx(data: bytes):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data)
        doc = Document(f.name)
        return "\n".join(p.text for p in doc.paragraphs)

def extract_eml(data: bytes):
    msg = email.message_from_bytes(data)
    for part in msg.walk():
        payload = part.get_payload(decode=True)
        if payload:
            return payload.decode(errors="ignore")
    return ""

def extract_msg_file(data: bytes):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data)
        return extract_msg.Message(f.name).body or ""

@app.post("/api/v1/hackrx/upload")
async def upload(file: UploadFile = File(...), questions: str = Form(...)):
    data = await file.read()
    name = file.filename.lower()

    if name.endswith(".pdf"):
        text = extract_pdf(data)
    elif name.endswith(".docx"):
        text = extract_docx(data)
    elif name.endswith(".eml"):
        text = extract_eml(data)
    elif name.endswith(".msg"):
        text = extract_msg_file(data)
    else:
        return {"error": "Unsupported format"}

    namespace = f"upload_{uuid4()}"
    chunks = get_chunks(clean_text(text))
    embed_to_pinecone(chunks, [{"text": c} for c in chunks], namespace)

    outputs = []
    for q in questions.split("\n"):
        lang = detect_language(q)
        en_q = translate_to_english(q) if lang != "English" else q
        ctx = get_relevant_context(en_q, namespace)
        ans = get_llm_answer(en_q, ctx)
        outputs.append({
            "question": q,
            "answer": translate_answer(ans, lang)
        })

    return {"answers": outputs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
@app.get("/health")
def health():
    return {"status": "ok"}
