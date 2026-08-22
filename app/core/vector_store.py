import os
import chromadb

CHROMA_DATA_PATH = os.path.join(os.getcwd(), "chroma_data")
os.makedirs(CHROMA_DATA_PATH, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_or_create_collection(name="rag_knowledge_base")


def add_document(doc_id: str, text: str):
    # Chroma standard local storage (uses built-in default embedding without downloads)
    collection.upsert(
        ids=[doc_id],
        documents=[text],
        metadatas=[{"doc_id": doc_id}]
    )


def query_documents(query_text: str, n_results: int = 2) -> str:
    # 1. First attempt direct Chroma search
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    if results and results.get("documents") and results["documents"][0]:
        return "\n".join(results["documents"][0])

    # 2. Fallback: Direct keyword match over all documents in collection
    all_docs = collection.get()
    if all_docs and all_docs.get("documents"):
        query_tokens = set(query_text.lower().split())
        matched = []
        for doc in all_docs["documents"]:
            doc_tokens = set(doc.lower().split())
            if query_tokens & doc_tokens:
                matched.append(doc)
        if matched:
            return "\n".join(matched[:n_results])

    return "No specific documents found."