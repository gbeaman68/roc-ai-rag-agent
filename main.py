# main.py  —  50 lines total
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain

# ---- 1. Load your vector store (exported from re:tune as `index.faiss`) ----
db = FAISS.load_local("index", OpenAIEmbeddings())

# ---- 2. Build a minimal RAG chain ----------------------------------------
model     = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = db.as_retriever(search_kwargs={"k": 4})
chain     = create_retrieval_chain(model, retriever)

# ---- 3. Define the FastAPI app -------------------------------------------
app = FastAPI()

class Q(BaseModel):
    question: str
    history: list[str] | None = []

@app.post("/ask")
def ask(q: Q):
    try:
        result = chain.invoke({"input": q.question, "chat_history": q.history})
        answer  = result["answer"]
        sources = [
            f"{i+1}. {doc.metadata['title']} ({doc.metadata.get('year','n.d.')})"
            for i, doc in enumerate(result["source_documents"])
        ]
        return {"answer": answer, "sources": "\n".join(sources)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
