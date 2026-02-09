from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import tiktoken
import os
from numpy.linalg import norm
from typing import List
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from docx import Document
import fitz  # PyMuPDF for PDFs
import zipfile
import chardet  # for encoding detection

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- CONFIG ---
MODEL = "gpt-4o"  # Updated from openai/gpt-5.2 to a valid model
EMBED_MODEL = "text-embedding-3-large"
DATA_FOLDER = "data"  # Folder with docx/pdf/txt files

api_key = os.getenv("AI_GATEWAY_API_KEY")
if not api_key:
    raise ValueError("❌ AI_GATEWAY_API_KEY not set. Add it to your .env file.")

client = OpenAI(
    api_key=api_key,
    base_url="https://ai-gateway.vercel.sh/v1"
)

app = FastAPI(title="Smart File Chatbot API", version="3.2")

# --- ENABLE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STATE ---
knowledge_base = []
chunk_embeddings = []
chat_history = []


# ==========================================================
# 📄 TEXT EXTRACTION HELPERS
# ==========================================================

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        print(f"⚠️ Error reading PDF {file_path}: {e}")
        return ""


def is_valid_docx(file_path):
    """Check if a DOCX file is valid and readable."""
    try:
        with zipfile.ZipFile(file_path, "r") as z:
            return "word/document.xml" in z.namelist()
    except:
        return False


def extract_text_from_docx(file_path: str) -> str:
    """Extract text safely from a DOCX file."""
    if not is_valid_docx(file_path):
        print(f"⚠️ Skipping invalid DOCX: {file_path}")
        return ""
    try:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        print(f"⚠️ Error reading DOCX {file_path}: {e}")
        return ""


def read_text_file_safely(file_path: str) -> str:
    """Read a text file with automatic encoding detection."""
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding", "utf-8")
            return raw.decode(encoding, errors="ignore")
    except Exception as e:
        print(f"⚠️ Could not read TXT {file_path}: {e}")
        return ""


def extract_text_from_pptx(file_path: str) -> str:
    """Extract text from a PowerPoint file."""
    try:
        prs = Presentation(file_path)
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
        return "\n".join(text_runs)
    except Exception as e:
        print(f"⚠️ Error reading PPTX {file_path}: {e}")
        return ""


def extract_text_from_csv(file_path: str) -> str:
    """Extract text from a CSV file by converting rows to strings."""
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    except Exception as e:
        print(f"⚠️ Error reading CSV {file_path}: {e}")
        return ""


def load_all_documents(folder_path: str) -> str:
    """Load and concatenate text from all supported files."""
    all_texts = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        ext = filename.lower()
        if ext.endswith(".pdf"):
            print(f"📘 Reading PDF: {filename}")
            all_texts.append(extract_text_from_pdf(file_path))
        elif ext.endswith(".docx"):
            print(f"📄 Reading DOCX: {filename}")
            all_texts.append(extract_text_from_docx(file_path))
        elif ext.endswith(".txt"):
            print(f"📜 Reading TXT: {filename}")
            all_texts.append(read_text_file_safely(file_path))
        elif ext.endswith(".pptx"):
            print(f"📊 Reading PPTX: {filename}")
            all_texts.append(extract_text_from_pptx(file_path))
        elif ext.endswith(".csv"):
            print(f"📈 Reading CSV: {filename}")
            all_texts.append(extract_text_from_csv(file_path))
        else:
            print(f"⚠️ Skipping unsupported file: {filename}")

    all_texts = [t for t in all_texts if t.strip()]
    return "\n\n".join(all_texts)


# ==========================================================
# 🧠 EMBEDDING + RETRIEVAL HELPERS
# ==========================================================

def chunk_text(text, max_tokens=700, overlap=150):
    """Smarter chunking with overlap for better context retention."""
    try:
        enc = tiktoken.encoding_for_model(MODEL)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        current_chunk = words[start:start + max_tokens]
        chunks.append(" ".join(current_chunk))
        start += max_tokens - overlap
    return chunks


def embed_texts(texts: List[str]):
    """Create embeddings for text chunks."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in response.data]


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def find_relevant_chunks(query, chunks, chunk_embeddings, top_n=8):
    """Find the most relevant text chunks using cosine similarity."""
    query_emb = embed_texts([query])[0]
    sims = [cosine_similarity(query_emb, emb) for emb in chunk_embeddings]
    
    # We use a combined score of similarity and a slight preference for longer chunks 
    # (which often contain more complete thoughts)
    scores = [s * (1 + 0.05 * np.log10(len(chunks[i]) + 1)) for i, s in enumerate(sims)]
    
    top_indices = np.argsort(scores)[-top_n:][::-1]
    
    # Filter out very low similarity results but keep a reasonable minimum
    return [chunks[i] for i in top_indices if sims[i] > 0.15]


def summarize_context(chunks, max_context_tokens=1500):
    """Condense multiple chunks if they exceed a certain length, otherwise return joined."""
    joined = "\n\n---\n\n".join(chunks)
    
    # Check token count (rough estimate: words * 1.3)
    estimated_tokens = len(joined.split()) * 1.3
    
    if estimated_tokens < max_context_tokens:
        return joined

    print(f"✂️ Context too long ({int(estimated_tokens)} tokens), summarizing...")
    summary_prompt = f"Summarize the following text into key bullet points for context retrieval:\n\n{joined}"
    summary = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    return summary.choices[0].message.content.strip()


def ask_gpt(question, context, chat_history):
    """Ask GPT using context and conversation memory."""
    system_prompt = """
    You are an intelligent assistant that answers questions ONLY using the provided context.
    - Combine information from multiple relevant sections.
    - Be factual, concise, and clear.
    - If unsure or the context lacks info, say: “I don’t know based on the available data.”
    """

    history_context = chat_history[-6:] if len(chat_history) > 6 else chat_history
    messages = [
        {"role": "system", "content": system_prompt},
        *history_context,
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=700,
    )

    answer = response.choices[0].message.content.strip()
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    return answer


# ==========================================================
# ⚡ AUTOLOAD ON STARTUP
# ==========================================================

@app.on_event("startup")
async def load_files_on_startup():
    """Load and embed DOCX/PDF/TXT files automatically at startup."""
    global knowledge_base, chunk_embeddings

    print("📂 Loading data files from ./data ...")
    if not os.path.exists(DATA_FOLDER):
        print("⚠️ No data folder found.")
        return

    full_text = load_all_documents(DATA_FOLDER)
    if not full_text.strip():
        print("⚠️ No readable text found in the data folder.")
        return

    print("🧩 Chunking and embedding text ...")
    knowledge_base = chunk_text(full_text)
    chunk_embeddings = embed_texts(knowledge_base)
    print(f"✅ Knowledge base ready: {len(knowledge_base)} chunks embedded.")


# ==========================================================
# 🌐 ROUTES ii
# ==========================================================

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(data: QuestionRequest):
    """Ask a question about the documents."""
    global knowledge_base, chunk_embeddings, chat_history

    if not knowledge_base:
        return {"error": "Knowledge base not initialized. Ensure DOCX/PDF/TXT files exist in ./data."}

    relevant_chunks = find_relevant_chunks(data.question, knowledge_base, chunk_embeddings)
    if not relevant_chunks:
        return {"answer": "Sorry, I couldn’t find relevant information in the documents."}

    summarized_context = summarize_context(relevant_chunks)
    answer = ask_gpt(data.question, summarized_context, chat_history)
    return {"answer": answer}


@app.get("/")
def root():
    return {"message": "🤖 Smart File Chatbot API v3.2 — trained on your DOCX/PDF/TXT files 🚀"}
