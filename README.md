# Agentic RAG Project

Syst�me de RAG agentique utilisant LangGraph pour combiner r�cup�ration vectorielle et base de donn�es priv�e.

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
venv\Scripts\activate  # Windows

# Installer les d�pendances
pip install -r requirements.txt
```

## Configuration

1. Copier `.env.example` vers `.env`
2. Configurer vos cl�s API et URLs de base de donn�es
3. Ajuster les param�tres dans `src/config/settings.py`

## Utilisation

```bash
# Tests
pytest tests/ -v

# Lancer l'agent
python src/agents/agentic_rag.py

# Mode d�veloppement
python scripts/dev.py run
```

## Architecture

- `src/agents/` : Agents RAG
- `src/database/` : Connecteurs base de donn�es
- `src/config/` : Configuration
- `data/` : Donn�es et bases vectorielles
- `tests/` : Tests unitaires

## D�veloppement avec Cline

Voir le guide complet dans le README pour l'utilisation optimale avec VS Code et Cline.
