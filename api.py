#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import query as rag_query
from agent import run_agent

app = FastAPI(title="AI Assessment API", version="1.0")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class AgentRequest(BaseModel):
    prompt: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rag/query")
def rag_endpoint(req: QueryRequest):
    try:
        return rag_query(req.question, top_k=req.top_k, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/plan")
def agent_endpoint(req: AgentRequest):
    try:
        return run_agent(req.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
