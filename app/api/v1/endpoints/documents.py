from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from app.tasks.worker import ingest_document_chunk, ingest_pdf_file

router = APIRouter()

class IngestTextRequest(BaseModel):
    doc_id: str
    content: str

@router.post("/ingest-text")
async def ingest_text_endpoint(payload: IngestTextRequest):
    task = ingest_document_chunk.delay(payload.doc_id, payload.content)
    return {"status": "Queued", "task_id": task.id, "type": "text"}

@router.post("/upload-pdf")
async def upload_pdf_endpoint(doc_id: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    task = ingest_pdf_file.delay(doc_id, contents)
    return {"status": "Queued", "task_id": task.id, "filename": file.filename, "type": "pdf"}