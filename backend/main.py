from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FastAPI setup
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
vector_store = None
chat_history = []  # Stores tuples of (question, answer)

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store, chat_history

    # Clear history when new PDF uploaded
    chat_history = []

    # Save temp PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Extract text from PDF
    reader = PdfReader(tmp_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # Remove temp file
    os.remove(tmp_path)

    return {"message": "PDF uploaded and processed successfully."}


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global vector_store, chat_history
    if vector_store is None:
        return JSONResponse(content={"error": "Please upload a PDF first."}, status_code=400)

    # Initialize LLM with Groq
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    # Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff",
        return_source_documents=False  # Set True if you want PDF chunks returned
    )

    # Combine history into prompt
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
    instruction = (
        "You are an assistant. Only answer questions using the content of the uploaded PDF. "
        "Do not make up answers. Stick strictly to the PDF."
    )
    final_question = f"{instruction}\n{history_text}\nQ: {question}"

    # Run QA
    answer = qa_chain.run(final_question)

    # Save to history
    chat_history.append((question, answer))

    return {"answer": answer}
