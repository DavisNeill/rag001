# Instructions Cline - Projet Agentic RAG

## Contexte
Syst�me RAG agentique avec LangGraph combinant r�cup�ration vectorielle et base de donn�es priv�e.

## Architecture
- Agent unique �voluant vers multi-agents
- LangGraph pour l'orchestration
- Chroma/Qdrant pour les vecteurs
- PostgreSQL/MongoDB pour donn�es priv�es

## Standards de Code
- Type hints obligatoires
- Docstrings Google style
- Tests unitaires syst�matiques
- Gestion d'erreurs robuste
- Logging informatif

## Priorit�s
1. S�curit� des donn�es
2. Performance des requ�tes
3. Maintenabilit�
4. Documentation

## Fichiers Cl�s
- `src/agents/agentic_rag.py` : Agent principal
- `src/config/settings.py` : Configuration
- `tests/` : Tests unitaires
- `requirements.txt` : D�pendances

## Commandes Fr�quentes
```bash
pytest tests/ -v
black src/
flake8 src/
python src/agents/agentic_rag.py
```
