# -*- coding: utf-8 -*-
"""
Système Agentic RAG avec Agent Unique
Architecture simple pour tests préliminaires
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from datetime import datetime

# Imports pour les composants LangChain et l'API Google
from config.settings import settings
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import os


class QueryType(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    MULTI_STEP = "multi_step"
    FACTUAL = "factual"

@dataclass
class AgentState:
    """État de l'agent à travers le workflow"""
    query: str
    query_type: QueryType
    retrieved_docs: List[Dict]
    db_results: List[Dict]
    response: str = ""
    confidence: float = 0.0
    sources: List[str] = None
    needs_clarification: bool = False
    iteration_count: int = 0

class AgenticRAGAgent:
    """Agent RAG Unique avec capacités avancées"""
    def __init__(
        self,
        vector_store_path: str = "./data/vector_db",
        db_connection_string: str = None
    ):
        """Initialise l'agent avec une connexion API (Gemini)"""

        # --- 1. Configuration de la Clé API ---
        if not settings.google_api_key:
            raise ValueError("La clé API Google (GOOGLE_API_KEY) n'est pas définie dans le fichier .env")
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key

        # --- 2. Initialisation des composants avec Google ---
        # Modèle de langage (LLM) avec Gemini
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True)

        # Modèle d'embeddings avec Google
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # --- 3. Le reste de la configuration ---
        self.vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=self.embeddings
        )

        self.private_db = self._init_private_db(db_connection_string)
        self.workflow = self._build_workflow()
    
    def _init_private_db(self, connection_string: str) -> Dict:
        """Initialise la connexion à la base de données privée"""
        # Simulation - remplacer par votre vraie DB
        return {
            "users": [{"id": 1, "name": "Jean", "role": "admin"}],
            "projects": [{"id": 1, "name": "Projet A", "status": "active"}],
            "documents": [{"id": 1, "title": "Doc confidentiel", "content": "..."}]
        }
    
    def _build_workflow(self) -> StateGraph:
        """Construit le workflow de l'agent"""
        workflow = StateGraph(AgentState)
        
        # Nœuds du workflow
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("retrieve_vector", self.retrieve_from_vector_store)
        workflow.add_node("query_database", self.query_private_database)
        workflow.add_node("validate_results", self.validate_results)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("self_evaluate", self.self_evaluate)
        
        # Définition des transitions
        workflow.set_entry_point("analyze_query")
        
        workflow.add_edge("analyze_query", "retrieve_vector")
        workflow.add_edge("retrieve_vector", "query_database")
        workflow.add_edge("query_database", "validate_results")
        
        # Branchement conditionnel après validation
        workflow.add_conditional_edges(
            "validate_results",
            self.should_regenerate,
            {
                "regenerate": "analyze_query",
                "continue": "generate_response"
            }
        )
        
        workflow.add_edge("generate_response", "self_evaluate")
        workflow.add_edge("self_evaluate", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    async def analyze_query(self, state: AgentState) -> AgentState:
        """Analyse la requête utilisateur"""
        query = state.query
        analysis_prompt = f"Analysez cette requête et déterminez: 1. Le type de requête (simple/complex/multi_step/factual), 2. Les entités importantes, 3. Le contexte nécessaire. Requête: {query}. Répondez en JSON avec: type, entities, context_needed"
        
        response = await self.llm.ainvoke(analysis_prompt)
        
        try:
            # CORRECTION : On accède à l'attribut .content de la réponse
            analysis = json.loads(response.content)
            state.query_type = QueryType(analysis.get("type", "simple"))
        except (json.JSONDecodeError, AttributeError):
            state.query_type = QueryType.SIMPLE
            
        return state
    
    async def retrieve_from_vector_store(self, state: AgentState) -> AgentState:
        """Récupère des documents du store vectoriel"""
        k = 5 if state.query_type == QueryType.SIMPLE else 10
        docs = self.vector_store.similarity_search(state.query, k=k)
        
        state.retrieved_docs = [{"content": doc.page_content, "metadata": doc.metadata, "source": doc.metadata.get("source", "unknown")} for doc in docs]
        return state
    
    async def query_private_database(self, state: AgentState) -> AgentState:
        """Interroge la base de données privée"""
        entities_prompt = f"Extrayez les entités pertinentes de cette requête pour une recherche en base: {state.query}. Entités possibles: users, projects, documents. Répondez uniquement avec les mots-clés séparés par des virgules."
        
        entities_response = await self.llm.ainvoke(entities_prompt)
        # CORRECTION : Cette partie était déjà correcte
        text_content = entities_response.content
        entities = [e.strip() for e in text_content.split(",")]
          
        db_results = []
        for table_name, records in self.private_db.items():
            for entity in entities:
                if entity.lower() in table_name.lower():
                    db_results.extend(records)
        
        state.db_results = db_results
        return state
    
    async def validate_results(self, state: AgentState) -> AgentState:
        """Valide la pertinence des résultats récupérés"""
        validation_prompt = f"Évaluez la pertinence de ces résultats pour la requête: Requête: {state.query}, Documents vectoriels: {len(state.retrieved_docs)} trouvés, Résultats DB: {len(state.db_results)} trouvés. Les résultats sont-ils suffisants pour répondre? (oui/non) Justification en une phrase."
        
        validation_response = await self.llm.ainvoke(validation_prompt)
        # CORRECTION : On accède à l'attribut .content de la réponse
        validation_text = validation_response.content
        
        has_content = len(state.retrieved_docs) > 0 or len(state.db_results) > 0
        validation_positive = "oui" in validation_text.lower()
        
        state.confidence = 0.8 if (has_content and validation_positive) else 0.3
        return state
    
    def should_regenerate(self, state: AgentState) -> str:
        """Décide s'il faut régénérer ou continuer"""
        if state.confidence < 0.5 and state.iteration_count < 2:
            state.iteration_count += 1
            return "regenerate"
        return "continue"
    
    async def generate_response(self, state: AgentState) -> AgentState:
        """Génère la réponse finale"""
        context_docs = "\n".join([f"Doc: {doc['content'][:200]}..." for doc in state.retrieved_docs[:3]])
        context_db = "\n".join([f"DB: {str(result)}" for result in state.db_results[:3]])
        generation_prompt = f"Répondez à cette question en utilisant uniquement les informations fournies: Question: {state.query}\n\nContexte Documents:\n{context_docs}\n\nContexte Base de Données:\n{context_db}\n\nInstructions: Réponse claire et précise, Citez vos sources, Si information manquante, le mentionner, Restez factuel."
        
        response = await self.llm.ainvoke(generation_prompt)
        # CORRECTION : On assigne le .content de la réponse, pas l'objet entier
        state.response = response.content
        
        state.sources = [doc["source"] for doc in state.retrieved_docs if doc["source"] not in (state.sources or [])]
        return state
    
    async def self_evaluate(self, state: AgentState) -> AgentState:
        """Auto-évaluation de la réponse"""
        eval_prompt = f"Évaluez cette réponse sur une échelle de 1-10: Question: {state.query}, Réponse: {state.response}. Critères: pertinence, complétude, factualité. Donnez juste le score (nombre entre 1-10)."
        
        try:
            score_response = await self.llm.ainvoke(eval_prompt)
            # CORRECTION : On accède à l'attribut .content de la réponse
            score_text = score_response.content
            score = float(score_text.strip())
            state.confidence = min(score / 10.0, 1.0)
        except (ValueError, AttributeError):
            state.confidence = 0.7
        
        return state
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Point d'entrée principal pour traiter une requête"""
        initial_state = AgentState(
            query=query,
            query_type=QueryType.SIMPLE,
            retrieved_docs=[],
            db_results=[],
            sources=[]
        )
        final_state = await self.workflow.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": f"session_{datetime.now().timestamp()}"}}
        )
        
        # S'assure que la réponse est bien une chaîne de caractères
        final_response = final_state.get("response", "")
        if isinstance(final_response, AIMessage):
            final_response = final_response.content

        return {
            "response": final_response,
            "confidence": final_state.get("confidence", 0.0),
            "sources": final_state.get("sources", []),
            "query_type": final_state.get("query_type", QueryType.SIMPLE).value,
            "retrieved_docs_count": len(final_state.get("retrieved_docs", [])),
            "db_results_count": len(final_state.get("db_results", []))
        }

async def main():
    """Exemple d'utilisation"""
    agent = AgenticRAGAgent(
        vector_store_path="./data/vector_db",
        db_connection_string="postgresql://user:pass@localhost/mydb"
    )
    test_queries = [
        "Quels sont les projets actifs?",
        "Résumez les documents confidentiels récents",
        "Qui sont les administrateurs du système?"
    ]
    
    for query in test_queries:
        print(f"\n🤖 Requête: {query}")
        result = await agent.process_query(query)
        
        print(f"📝 Réponse: {result['response']}")
        print(f"🎯 Confiance: {result['confidence']:.2f}")
        print(f"📚 Sources: {result['sources']}")
        print(f"📊 Docs trouvés: {result['retrieved_docs_count']}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())