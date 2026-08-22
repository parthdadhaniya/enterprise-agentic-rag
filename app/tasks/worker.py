import os
import io
from celery import Celery
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.vector_store import collection
from dotenv import load_dotenv

load_dotenv()

celery_app = Celery("rag_tasks", broker=os.getenv("REDIS_URL"), backend=os.getenv("REDIS_URL"))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)


@celery_app.task(name="tasks.ingest_document_chunk")
def ingest_document_chunk(doc_id: str, text: str):
    try:
        chunks = text_splitter.split_text(text)
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )
        return {"status": "SUCCESS", "doc_id": doc_id, "total_chunks": len(chunks)}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


@celery_app.task(name="tasks.ingest_pdf_file")
def ingest_pdf_file(doc_id: str, file_bytes: bytes):
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n[Page {page_num + 1}]\n" + text

        chunks = text_splitter.split_text(full_text)
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )
        return {"status": "SUCCESS", "doc_id": doc_id, "total_chunks": len(chunks)}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}