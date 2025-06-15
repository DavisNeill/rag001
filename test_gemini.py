# -*- coding: utf-8 -*-
import asyncio
import os
from config.settings import settings
from langchain_google_genai import ChatGoogleGenerativeAI

async def run_simple_test():
    """Teste une connexion simple à l'API Gemini."""
    print("1. Vérification de la clé API...")
    if not settings.google_api_key:
        print("ERREUR : La clé API Google n'est pas trouvée dans .env !")
        return

    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    print("   Clé API chargée.")

    print("\n2. Initialisation du modèle Gemini...")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        print("   Modèle initialisé avec succès.")
    except Exception as e:
        print(f"ERREUR lors de l'initialisation : {e}")
        return

    print("\n3. Envoi d'une requête de test à l'API...")
    try:
        response = await llm.ainvoke("Bonjour, dis juste 'test OK'")
        print("\n✅ SUCCÈS ! Réponse de l'API reçue :")
        print(response.content)
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'appel à l'API : {e}")

if __name__ == "__main__":
    asyncio.run(run_simple_test())