#!/usr/bin/env python3
"""
Script d'installation automatique pour le projet Agentic RAG
Utiliser : python setup_project.py
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def create_directory_structure():
    """Crée la structure de dossiers du projet"""
    directories = [
        "src/agents",
        "src/database", 
        "src/utils",
        "src/config",
        "src/api",
        "data/documents",
        "data/vector_db",
        "data/private_db",
        "tests",
        "notebooks",
        "scripts",
        "prompts",
        ".vscode",
        ".cline"
    ]
    
    print("Création de la structure de dossiers...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Créer __init__.py pour les packages Python
        if directory.startswith("src/"):
            (Path(directory) / "__init__.py").touch()
    
    print("Structure créée avec succès!")

def create_requirements_file():
    """Crée le fichier requirements.txt"""
    requirements = """# Core LangChain/LangGraph
langchain==0.1.0
langgraph==0.0.40
langchain-community==0.0.20
langchain-core==0.1.23

# Vector Stores
chromadb==0.4.22
qdrant-client==1.7.0

# Embeddings & LLMs
sentence-transformers==2.2.2
ollama==0.1.7
openai==1.12.0

# Database
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
pymongo==4.6.1

# Utilities
python-dotenv==1.0.0
pydantic==2.5.3
pydantic-settings==2.1.0
fastapi==0.108.0
uvicorn==0.25.0
pandas==2.1.4
numpy==1.24.3

# CLI
click==8.1.7

# Development
pytest==7.4.4
pytest-cov==4.1.0
black==23.12.1
flake8==7.0.0
pre-commit==3.6.0
jupyter==1.0.0

# Monitoring
wandb==0.16.2
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("requirements.txt créé!")

def create_env_file():
    """Crée le fichier .env template"""
    env_content = """# LLM Configuration
OPENAI_API_KEY=your_openai_key_here
OLLAMA_BASE_URL=http://localhost:11434

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/your_db
MONGODB_URL=mongodb://localhost:27017/your_db

# Vector Store
CHROMA_PERSIST_DIRECTORY=./data/vector_db
QDRANT_URL=http://localhost:6333

# Application
DEBUG=True
LOG_LEVEL=INFO
MAX_ITERATIONS=3
CONFIDENCE_THRESHOLD=0.7
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    print(".env template créé!")

def create_vscode_config():
    """Crée la configuration VS Code"""
    settings = {
        "python.defaultInterpreterPath": "./venv/bin/python",
        "python.linting.enabled": True,
        "python.linting.flake8Enabled": True,
        "python.formatting.provider": "black",
        "python.testing.pytestEnabled": True,
        "python.testing.pytestArgs": ["tests"],
        "files.exclude": {
            "**/__pycache__": True,
            "**/*.pyc": True,
            ".pytest_cache": True,
            "*.egg-info": True
        },
        "cline.maxTokens": 100000,
        "cline.model": "claude-3-5-sonnet-20241022"
    }
    
    with open(".vscode/settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Agentic RAG",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/src/agents/agentic_rag.py",
                "console": "integratedTerminal",
                "env": {
                    "PYTHONPATH": "${workspaceFolder}/src"
                },
                "cwd": "${workspaceFolder}"
            }
        ]
    }
    
    with open(".vscode/launch.json", "w") as f:
        json.dump(launch_config, f, indent=2)
    
    print("Configuration VS Code créée!")

def create_gitignore():
    """Crée le fichier .gitignore"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv/
*.egg-info/
dist/
build/

# IDE
.vscode/settings.json
.idea/
*.swp
*.swo

# Environment
.env
.env.local

# Data
data/vector_db/
data/private_db/
*.db
*.sqlite

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Models
models/
*.bin
*.safetensors

# API Keys
*.key
secrets/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print(".gitignore créé!")

def create_basic_files():
    """Crée les fichiers de base du projet"""
    
    # README.md
    readme_content = """# Agentic RAG Project

Système de RAG agentique utilisant LangGraph pour combiner récupération vectorielle et base de données privée.

## Installation

```bash
# Cloner et configurer
git clone <your-repo>
cd agentic-rag-project

# Setup automatique
python setup_project.py

# Activer l'environnement
source venv/bin/activate  # Linux/Mac
# ou
venv\\Scripts\\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Configuration

1. Copier `.env.example` vers `.env`
2. Configurer vos clés API et URLs de base de données
3. Ajuster les paramètres dans `src/config/settings.py`

## Utilisation

```bash
# Tests
pytest tests/ -v

# Lancer l'agent
python src/agents/agentic_rag.py

# Mode développement
python scripts/dev.py run
```

## Architecture

- `src/agents/` : Agents RAG
- `src/database/` : Connecteurs base de données
- `src/config/` : Configuration
- `data/` : Données et bases vectorielles
- `tests/` : Tests unitaires

## Développement avec Cline

Voir le guide complet dans le README pour l'utilisation optimale avec VS Code et Cline.
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    # Configuration settings
    settings_content = '''"""Configuration globale du projet"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Configuration de l'application"""
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama2"
    embedding_model: str = "nomic-embed-text"
    
    # Database
    database_url: Optional[str] = None
    mongodb_url: Optional[str] = None
    
    # Vector Store
    chroma_persist_directory: str = "./data/vector_db"
    qdrant_url: Optional[str] = None
    
    # Application
    debug: bool = True
    log_level: str = "INFO"
    max_iterations: int = 3
    confidence_threshold: float = 0.7
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
'''
    
    with open("src/config/settings.py", "w") as f:
        f.write(settings_content)
    
    # Test basique
    test_content = '''"""Tests de base pour l'agent RAG"""

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
'''
    
    with open("tests/test_agent.py", "w") as f:
        f.write(test_content)
    
    # Instructions Cline
    cline_instructions = '''# Instructions Cline - Projet Agentic RAG

## Contexte
Système RAG agentique avec LangGraph combinant récupération vectorielle et base de données privée.

## Architecture
- Agent unique évoluant vers multi-agents
- LangGraph pour l'orchestration
- Chroma/Qdrant pour les vecteurs
- PostgreSQL/MongoDB pour données privées

## Standards de Code
- Type hints obligatoires
- Docstrings Google style
- Tests unitaires systématiques
- Gestion d'erreurs robuste
- Logging informatif

## Priorités
1. Sécurité des données
2. Performance des requêtes
3. Maintenabilité
4. Documentation

## Fichiers Clés
- `src/agents/agentic_rag.py` : Agent principal
- `src/config/settings.py` : Configuration
- `tests/` : Tests unitaires
- `requirements.txt` : Dépendances

## Commandes Fréquentes
```bash
pytest tests/ -v
black src/
flake8 src/
python src/agents/agentic_rag.py
```
'''
    
    with open(".cline/instructions.md", "w") as f:
        f.write(cline_instructions)
    
    print("Fichiers de base créés!")

def setup_virtual_environment():
    """Configure l'environnement virtuel"""
    if not Path("venv").exists():
        print("Création de l'environnement virtuel...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("Environnement virtuel créé!")
    else:
        print("Environnement virtuel existe déjà!")

def create_dev_scripts():
    """Crée les scripts de développement"""
    dev_script = '''#!/usr/bin/env python3
"""Scripts utilitaires de développement"""

import click
import subprocess
import sys
import os
from pathlib import Path

@click.group()
def cli():
    """Utilitaires de développement Agentic RAG"""
    pass

@cli.command()
def setup():
    """Installation complète"""
    click.echo("Installation des dépendances...")
    
    # Vérifier l'environnement virtuel
    if not os.path.exists("venv"):
        click.echo("Créez d'abord l'environnement virtuel : python -m venv venv")
        return
    
    # Installer les dépendances
    if sys.platform == "win32":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    subprocess.run([pip_cmd, "install", "-r", "requirements.txt"])
    
    # Pre-commit hooks (optionnel)
    try:
        subprocess.run([pip_cmd, "install", "pre-commit"])
        subprocess.run(["pre-commit", "install"])
        click.echo("Pre-commit hooks installés")
    except:
        click.echo("Pre-commit hooks non installés (optionnel)")
    
    click.echo("Setup terminé !")

@cli.command()
def test():
    """Lancer tous les tests"""
    click.echo("Tests en cours...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--cov=src"])

@cli.command()  
def lint():
    """Linting et formatage"""
    click.echo("Formatage du code...")
    subprocess.run([sys.executable, "-m", "black", "src/", "tests/"])
    subprocess.run([sys.executable, "-m", "flake8", "src/"])

@cli.command()
def run():
    """Lancer l'agent en mode interactif"""
    click.echo("Démarrage de l'agent...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").absolute())
    subprocess.run([sys.executable, "src/agents/agentic_rag.py"], env=env)

@cli.command()
def serve():
    """Lancer l'API FastAPI"""
    click.echo("Démarrage du serveur API...")
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "src.api.main:app", 
        "--reload", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ])

@cli.command()
def notebook():
    """Lancer Jupyter pour expérimentation"""
    click.echo("Démarrage de Jupyter...")
    subprocess.run([sys.executable, "-m", "jupyter", "lab", "notebooks/"])

@cli.command()
def ollama_setup():
    """Vérifier et configurer Ollama"""
    click.echo("Configuration d'Ollama...")
    
    # Vérifier si Ollama est installé
    try:
        result = subprocess.run(["ollama", "version"], capture_output=True, text=True)
        click.echo(f"Ollama installé : {result.stdout.strip()}")
    except FileNotFoundError:
        click.echo("Ollama non trouvé. Installez-le depuis https://ollama.ai")
        return
    
    # Télécharger les modèles nécessaires
    models = ["llama2", "nomic-embed-text"]
    for model in models:
        click.echo(f"Téléchargement de {model}...")
        subprocess.run(["ollama", "pull", model])
    
    click.echo("Ollama configuré avec succès!")

@cli.command()
def clean():
    """Nettoyer les fichiers temporaires"""
    click.echo("Nettoyage...")
    
    import shutil
    
    # Dossiers à nettoyer
    to_clean = [
        "__pycache__",
        ".pytest_cache", 
        ".coverage",
        "htmlcov",
        "dist",
        "build",
        "*.egg-info"
    ]
    
    for pattern in to_clean:
        for path in Path(".").rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                click.echo(f"Supprimé : {path}")
            elif path.is_file():
                path.unlink()
                click.echo(f"Supprimé : {path}")
    
    click.echo("Nettoyage terminé!")

if __name__ == "__main__":
    cli()
'''
    
    with open("scripts/dev.py", "w") as f:
        f.write(dev_script)
    
    # Test d'installation
    test_install_script = '''#!/usr/bin/env python3
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
        print("Ollama : Timeout (vérifiez qu'Ollama est démarré)")
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
        print("Agent RAG : Importé avec succès")
    except Exception as e:
        print(f"Agent RAG : {str(e)[:50]}...")
    
    print("=" * 50)
    print("Tests terminés !")
    print("\n Prochaines étapes :")
    print("1. Configurez vos clés API dans .env")
    print("2. Démarrez Ollama si pas fait : ollama serve")
    print("3. Testez avec : python scripts/dev.py run")
    print("4. Développez avec Cline dans VS Code")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_environment())
'''
    
    with open("test_installation.py", "w") as f:
        f.write(test_install_script)
    
    # API FastAPI basique
    api_main = '''"""API FastAPI pour le système Agentic RAG"""

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
'''
    
    with open("src/api/main.py", "w") as f:
        f.write(api_main)
    
    # Rendre les scripts exécutables
    os.chmod("scripts/dev.py", 0o755)
    os.chmod("test_installation.py", 0o755)
    
    print("Scripts de développement créés!")

def create_example_notebook():
    """Crée un notebook d'exemple pour l'expérimentation"""
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Agentic RAG - Notebook d'Expérimentation\\n",
    "\\n",
    "Ce notebook permet de tester et expérimenter avec le système Agentic RAG."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Imports et configuration\\n",
    "import sys\\n",
    "from pathlib import Path\\n",
    "\\n",
    "# Ajouter src au path\\n",
    "sys.path.append(str(Path('..') / 'src'))\\n",
    "\\n",
    "import asyncio\\n",
    "from agents.agentic_rag import AgenticRAGAgent\\n",
    "from config.settings import settings\\n",
    "\\n",
    "print(\\" Configuration chargée\\")\\n",
    "print(f\\"Debug mode: {settings.debug}\\")\\n",
    "print(f\\"LLM Model: {settings.llm_model}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialiser l'agent\\n",
    "agent = AgenticRAGAgent()\\n",
    "print(\\"Agent initialisé\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test de requête simple\\n",
    "async def test_query(query_text):\\n",
    "    result = await agent.process_query(query_text)\\n",
    "    \\n",
    "    print(f\\"Requête: {query_text}\\")\\n",
    "    print(f\\"Réponse: {result['response']}\\")\\n",
    "    print(f\\"Confiance: {result['confidence']:.2f}\\")\\n",
    "    print(f\\"Sources: {result['sources']}\\")\\n",
    "    print(f\\"Docs: {result['retrieved_docs_count']}, DB: {result['db_results_count']}\\")\\n",
    "    \\n",
    "    return result\\n",
    "\\n",
    "# Tester\\n",
    "result = await test_query(\\"Quels sont les projets actifs?\\")\\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tests multiples\\n",
    "test_queries = [\\n",
    "    \\"Bonjour\\",\\n",
    "    \\"Quels sont les projets actifs?\\",\\n",
    "    \\"Résumez les documents récents\\",\\n",
    "    \\"Qui sont les administrateurs?\\"\\n",
    "]\\n",
    "\\n",
    "results = []\\n",
    "for query in test_queries:\\n",
    "    print(f\\"\\\\n{'='*50}\\")\\n",
    "    result = await test_query(query)\\n",
    "    results.append(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyse des performances\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "\\n",
    "# Créer un DataFrame avec les résultats\\n",
    "df = pd.DataFrame([\\n",
    "    {\\n",
    "        'query': query,\\n",
    "        'confidence': result['confidence'],\\n",
    "        'docs_count': result['retrieved_docs_count'],\\n",
    "        'db_count': result['db_results_count']\\n",
    "    }\\n",
    "    for query, result in zip(test_queries, results)\\n",
    "])\\n",
    "\\n",
    "# Graphique des scores de confiance\\n",
    "plt.figure(figsize=(10, 6))\\n",
    "plt.bar(range(len(df)), df['confidence'])\\n",
    "plt.xlabel('Requêtes')\\n",
    "plt.ylabel('Score de Confiance')\\n",
    "plt.title('Scores de Confiance par Requête')\\n",
    "plt.xticks(range(len(df)), [q[:20] + '...' if len(q) > 20 else q for q in df['query']], rotation=45)\\n",
    "plt.tight_layout()\\n",
    "plt.show()\\n",
    "\\n",
    "print(df)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    with open("notebooks/agentic_rag_experiments.ipynb", "w") as f:
        f.write(notebook_content)
    
    print("Notebook d'expérimentation créé!")

def main():
    """Fonction principale d'installation"""
    print("Setup Automatique - Projet Agentic RAG")
    print("=" * 50)
    
    try:
        # Étapes d'installation
        create_directory_structure()
        create_requirements_file()
        create_env_file()
        create_vscode_config()
        create_gitignore()
        create_basic_files()
        setup_virtual_environment()
        create_dev_scripts()
        create_example_notebook()
        
        print("\n" + "=" * 50)
        print("Installation terminée avec succès !")
        print("\n Prochaines étapes :")
        print("1. Activez l'environnement virtuel :")
        print("   source venv/bin/activate  # Linux/Mac")
        print("   venv\\Scripts\\activate     # Windows")
        print("\n2. Installez les dépendances :")
        print("   python scripts/dev.py setup")
        print("\n3. Configurez vos clés API dans .env")
        print("\n4. Testez l'installation :")
        print("   python test_installation.py")
        print("\n5. Ouvrez VS Code et installez Cline")
        print("\n6. Commencez le développement !")
        print("\n Ressources :")
        print("- README.md : Documentation complète")
        print("- notebooks/ : Expérimentation Jupyter") 
        print("- scripts/dev.py : Commandes utilitaires")
        print("- .cline/ : Instructions pour Cline")
        
    except Exception as e:
        print(f"Erreur lors de l'installation : {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)