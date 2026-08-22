import os
from typing import TypedDict, Annotated, Sequence, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.core.vector_store import collection
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    intent: str
    context: str
    sources: List[dict]
    grade: str


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


def classify_intent(state: AgentState):
    query = state["messages"][-1].content.strip().lower()
    greetings = {"hi", "hello", "hey", "hola", "namaste", "good morning", "good evening", "who are you"}

    if query in greetings or (len(query.split()) <= 2 and any(g in query for g in ["hi", "hello", "hey"])):
        return {"intent": "GENERAL_QUERY"}

    return {"intent": "DOCUMENT_QUERY"}


def rag_retriever_node(state: AgentState):
    query = state["messages"][-1].content
    results = collection.query(query_texts=[query], n_results=3)

    docs = []
    sources = []
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

    context = "\n---\n".join(docs) if docs else "No specific documents found."
    grade = "RELEVANT" if docs else "NOT_RELEVANT"
    return {"context": context, "sources": sources, "grade": grade}


def generate_answer(state: AgentState):
    messages = list(state["messages"])
    query_text = messages[-1].content

    # Format message history into prompt context
    if state.get("intent") == "DOCUMENT_QUERY" and state.get("context") and state[
        "context"] != "No specific documents found.":
        system_instruction = (
            "You are an enterprise AI assistant with access to document context and previous conversation history.\n"
            "Answer the question accurately based on the context and conversation flow.\n\n"
            f"Document Context:\n{state['context']}"
        )
        conversation_payload = [HumanMessage(content=system_instruction)] + messages
        res = llm.invoke(conversation_payload)
    else:
        res = llm.invoke(messages)

    return {"messages": [res]}


def route_intent(state: AgentState):
    return "rag_retriever" if state["intent"] == "DOCUMENT_QUERY" else "generate_answer"


# Construct State Graph with Checkpointer
workflow = StateGraph(AgentState)
workflow.add_node("classifier", classify_intent)
workflow.add_node("rag_retriever", rag_retriever_node)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("classifier")
workflow.add_conditional_edges("classifier", route_intent, {
    "rag_retriever": "rag_retriever",
    "generate_answer": "generate_answer"
})
workflow.add_edge("rag_retriever", "generate_answer")
workflow.add_edge("generate_answer", END)

# In-memory/Redis checkpointer
checkpointer = MemorySaver()
agent_app = workflow.compile(checkpointer=checkpointer)