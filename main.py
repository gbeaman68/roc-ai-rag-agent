# main.py  – FastAPI RAG micro-service
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# -------------------------------------------------
# 1.  Vector-store (FAISS) – load or fall back
# -------------------------------------------------
INDEX_PATH = "index"          # folder created in the repo
INDEX_NAME = "index"          # <index_name>.faiss / <index_name>.pkl

def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    try:
        # Try to load a real index if it exists
        return FAISS.load_local(INDEX_PATH, embeddings, index_name=INDEX_NAME)
    except Exception:
        # Otherwise spin up a placeholder so the service still starts
        return FAISS.from_texts(["placeholder"], embeddings)

vectorstore = load_vectorstore()
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------------------------
# 2.  Retrieval-augmented generation chain
# -------------------------------------------------
PROMPT_TMPL = """
You are a helpful assistant. Use the context below to answer the question.

<context>
{context}
</context>

Question: {question}
Answer:
""".strip()

prompt   = ChatPromptTemplate.from_template(PROMPT_TMPL)
llm      = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = prompt | llm           # “combine_docs_chain” in LC docs

# New LC 0.2.x signature → no ‘return_source_documents’ kwarg
chain = create_retrieval_chain(retriever, qa_chain)

# -------------------------------------------------
# 3.  FastAPI wiring
# -------------------------------------------------
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Query):
    """
    POST {"question": "..."}  →  {"answer": "...", "context": [...]}
    """
    try:
        result = chain.invoke({"question": q.question})
        return {
            "answer":  result["answer"],
            "context": result["context"],   # list of strings (docs merged by LC)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
