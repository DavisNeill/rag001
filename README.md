# Agentic RAG Project

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
venv\Scripts\activate  # Windows

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
