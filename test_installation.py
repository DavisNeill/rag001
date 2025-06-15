# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Script de test de l'installation"""

import asyncio
import sys
import os
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).parent / "src"))

async def test_environment():
    """Test complet de l'environnement"""
    
    print("Test de l'environnement Agentic RAG...")
    print("=" * 50)
    
    # Test 1: Imports Python
    try:
        import langchain
        import langgraph
        import chromadb
        print("LangChain/LangGraph : OK")
    except ImportError as e:
        print(f"Import Error : {e}")
        return False
    
    # Test 2: Configuration
    try:
        from src.config.settings import settings
        print(f"Configuration : OK (Debug: {settings.debug})")
    except Exception as e:
        print(f"Configuration Error : {e}")
    
    # Test 3: Ollama (si disponible)
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama2", base_url=settings.ollama_base_url)
        
        # Test simple avec timeout
        response = await asyncio.wait_for(
            llm.ainvoke("Dis juste 'test ok'"), 
            timeout=10.0
        )
        print(f"Ollama : OK - {response[:30]}...")
        
    except asyncio.TimeoutError:
        print("Ollama : Timeout (v�rifiez qu'Ollama est d�marr�)")
    except Exception as e:
        print(f"Ollama : {str(e)[:50]}...")
    
    # Test 4: Base vectorielle
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import OllamaEmbeddings
        
        # Test embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        test_embed = await asyncio.wait_for(
            embeddings.aembed_query("test"), 
            timeout=10.0
        )
        print(f"Embeddings : OK (dim: {len(test_embed)})")
        
        # Test Chroma
        vector_store = Chroma(
            persist_directory="./test_chroma",
            embedding_function=embeddings
        )
        print("Chroma : OK")
        
    except Exception as e:
        print(f"Vector Store : {str(e)[:50]}...")
    
    # Test 5: Structure des dossiers
    required_dirs = [
        "src/agents", "src/config", "src/database",
        "data/vector_db", "tests", "notebooks"
    ]
    
    missing_dirs = [d for d in required_dirs if not Path(d).exists()]
    if missing_dirs:
        print(f"Dossiers manquants : {missing_dirs}")
    else:
        print("Structure des dossiers : OK")
    
    # Test 6: Agent RAG (si disponible)
    try:
        from src.agents.agentic_rag import AgenticRAGAgent
        agent = AgenticRAGAgent()
        print("Agent RAG : Import� avec succ�s")
    except Exception as e:
        print(f"Agent RAG : {str(e)[:50]}...")
    
    print("=" * 50)
    print("Tests termin�s !")
    print("""
 Prochaines �tapes :")
    print("1. Configurez vos cl�s API dans .env")
    print("2. D�marrez Ollama si pas fait : ollama serve")
    print("3. Testez avec : python scripts/dev.py run")
    print("4. D�veloppez avec Cline dans VS Code""")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_environment())
