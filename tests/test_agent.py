"""Tests de base pour l'agent RAG"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

# Ajouter le chemin src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.agentic_rag import AgenticRAGAgent, AgentState, QueryType

class TestAgenticRAGAgent:
    """Tests pour l'agent RAG"""
    
    @pytest.fixture
    def agent(self):
        """Fixture pour créer un agent de test"""
        return AgenticRAGAgent(
            vector_store_path="./test_db",
            llm_model="llama2"
        )
    
    def test_agent_initialization(self, agent):
        """Test de l'initialisation de l'agent"""
        assert agent is not None
        assert agent.llm is not None
        assert agent.vector_store is not None
    
    @pytest.mark.asyncio
    async def test_analyze_query(self, agent):
        """Test de l'analyse de requête"""
        state = AgentState(
            query="Quels sont les projets actifs?",
            query_type=QueryType.SIMPLE,
            retrieved_docs=[],
            db_results=[]
        )
        
        # Mock du LLM pour éviter les appels réels
        agent.llm.ainvoke = AsyncMock(return_value='{"type": "simple", "entities": ["projets"], "context_needed": "database"}')
        
        result = await agent.analyze_query(state)
        
        assert result.query_type == QueryType.SIMPLE
        agent.llm.ainvoke.assert_called_once()
    
    def test_query_classification(self):
        """Test de la classification des requêtes"""
        queries = [
            ("Bonjour", QueryType.SIMPLE),
            ("Analyse détaillée des performances", QueryType.COMPLEX),
            ("Qui est responsable et quel est le budget?", QueryType.MULTI_STEP)
        ]
        
        # Tests logiques sans appels LLM
        for query, expected_type in queries:
            # Logique de classification simplifiée pour test
            if "et" in query.lower() or "?" in query:
                assert expected_type in [QueryType.MULTI_STEP, QueryType.COMPLEX]
            else:
                assert expected_type == QueryType.SIMPLE

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
