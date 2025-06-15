# Instructions Cline - Projet Agentic RAG

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
