"""API FastAPI pour le système Agentic RAG"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import sys
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

# Créer l'app FastAPI
app = FastAPI(
    title="Agentic RAG API",
    description="API pour système RAG agentique avec LangGraph",
    version="1.0.0"
)

# Modèles Pydantic
class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    max_iterations: Optional[int] = None

class QueryResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    query_type: str
    retrieved_docs_count: int
    db_results_count: int
    processing_time: float

# Variable globale pour l'agent (lazy loading)
agent = None

async def get_agent():
    """Obtenir une instance de l'agent (lazy loading)"""
    global agent
    if agent is None:
        try:
            from agents.agentic_rag import AgenticRAGAgent
            agent = AgenticRAGAgent()
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur d'initialisation de l'agent : {str(e)}"
            )
    return agent

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "Agentic RAG API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Vérification de santé de l'API"""
    try:
        agent_instance = await get_agent()
        return {
            "status": "healthy",
            "agent": "initialized",
            "config": {
                "debug": settings.debug,
                "llm_model": settings.llm_model,
                "confidence_threshold": settings.confidence_threshold
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e)
        }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Traiter une requête avec l'agent RAG"""
    
    import time
    start_time = time.time()
    
    try:
        agent_instance = await get_agent()
        
        # Traiter la requête
        result = await agent_instance.process_query(request.query)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            response=result["response"],
            confidence=result["confidence"],
            sources=result["sources"],
            query_type=result["query_type"],
            retrieved_docs_count=result["retrieved_docs_count"],
            db_results_count=result["db_results_count"],
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement : {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Statistiques du système"""
    try:
        agent_instance = await get_agent()
        
        # Statistiques basiques
        return {
            "agent_status": "active",
            "vector_store_path": settings.chroma_persist_directory,
            "llm_model": settings.llm_model,
            "confidence_threshold": settings.confidence_threshold,
            "max_iterations": settings.max_iterations
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur stats : {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=settings.api_host, 
        port=settings.api_port,
        reload=settings.debug
    )
