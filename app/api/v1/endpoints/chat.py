import json
import asyncio
from typing import Optional
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from app.agents.graph import agent_app, llm
from app.core.vector_store import collection

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default_session"


def parse_llm_content(raw_content) -> str:
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in raw_content]
        return "\n".join(text_parts)
    return str(raw_content)


@router.post("")
async def chat_endpoint(payload: ChatRequest):
    config = {"configurable": {"thread_id": payload.session_id}}
    state = {
        "messages": [HumanMessage(content=payload.query)],
        "intent": "",
        "context": "",
        "sources": [],
        "grade": ""
    }
    result = await agent_app.ainvoke(state, config=config)
    clean_text = parse_llm_content(result["messages"][-1].content)

    return {
        "response": clean_text,
        "session_id": payload.session_id,
        "intent": result.get("intent"),
        "relevance_grade": result.get("grade", "N/A"),
        "sources": result.get("sources", [])
    }


@router.post("/stream")
async def chat_stream_endpoint(payload: ChatRequest):
    config = {"configurable": {"thread_id": payload.session_id}}
    query = payload.query.strip()

    current_state = await agent_app.aget_state(config)
    past_messages = current_state.values.get("messages", []) if current_state else []

    results = collection.query(query_texts=[query], n_results=3)
    docs, sources = [], []
    if results and results.get("documents") and results["documents"][0]:
        for i, text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            distance = results["distances"][0][i] if results.get("distances") else 0.0
            docs.append(text)
            sources.append({
                "doc_id": meta.get("doc_id", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "distance": round(distance, 4)
            })

    context = "\n---\n".join(docs) if docs else ""
    grade = "RELEVANT" if docs else "NOT_RELEVANT"

    async def event_generator():
        meta_payload = {
            "type": "metadata",
            "session_id": payload.session_id,
            "relevance_grade": grade,
            "sources": sources
        }
        yield f"data: {json.dumps(meta_payload)}\n\n"

        prompt_messages = list(past_messages)
        if docs:
            prompt_messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"))
        else:
            prompt_messages.append(HumanMessage(content=query))

        full_response_text = ""
        async for chunk in llm.astream(prompt_messages):
            token_text = chunk.content if isinstance(chunk.content, str) else ""
            if token_text:
                full_response_text += token_text
                yield f"data: {json.dumps({'type': 'token', 'text': token_text})}\n\n"
                await asyncio.sleep(0.01)

        await agent_app.aupdate_state(
            config,
            {"messages": [HumanMessage(content=query), AIMessage(content=full_response_text)]},
            as_node="generate_answer"
        )

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )