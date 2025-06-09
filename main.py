# main.py  – ROC-AI Retrieval + Citation micro-service
# -----------------------------------------------
# FastAPI + LangChain 0.2.x (community split) + FAISS
# -----------------------------------------------

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain imports (v0.2.x)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain

# -------------------------------------------------------------------
# 1. Load (or create) the vector store
# -------------------------------------------------------------------
INDEX_PATH = "index"   # folder holding index.faiss / index.pkl

embeddings = OpenAIEmbeddings()
try:
    # If you committed an index folder, load it (allow pickle deserialization)
    db = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception:
    # Fallback: boot an empty store so the app still runs
    db = FAISS.from_texts(["placeholder"], embeddings)

# -------------------------------------------------------------------
# 2. Build a minimal RAG chain
# -------------------------------------------------------------------
llm       = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = db.as_retriever(search_kwargs={"k": 4})
chain     = create_retrieval_chain(llm, retriever, return_source_documents=True)

# -------------------------------------------------------------------
# 3. FastAPI app
# -------------------------------------------------------------------
app = FastAPI()

class Q(BaseModel):
    question: str
    history: list[str] | None = []

@app.post("/ask")
def ask(q: Q):
    """Answer a question and return APA-style sources."""
    try:
        result  = chain.invoke({"input": q.question, "chat_history": q.history})
        answer  = result["answer"]
        sources = [
            f"{i+1}. {d.metadata.get('title', 'Untitled')} "
            f"({d.metadata.get('year', 'n.d.')})"
            for i, d in enumerate(result.get("source_documents", []))
        ]
        return {"answer": answer, "sources": "\n".join(sources)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
