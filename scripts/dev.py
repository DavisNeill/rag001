# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Scripts utilitaires de d�veloppement"""

import click
import subprocess
import sys
import os
from pathlib import Path

@click.group()
def cli():
    """Utilitaires de d�veloppement Agentic RAG"""
    pass

@cli.command()
def setup():
    """Installation compl�te"""
    click.echo("Installation des d�pendances...")
    
    # V�rifier l'environnement virtuel
    if not os.path.exists("venv"):
        click.echo("Cr�ez d'abord l'environnement virtuel : python -m venv venv")
        return
    
    # Installer les d�pendances
    if sys.platform == "win32":
        pip_cmd = r"venv\Scripts\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    # Modifiez la ligne pour qu'elle ressemble à ceci :
    subprocess.run([pip_cmd, "install", "--no-cache-dir", "--upgrade", "-r", "requirements.txt"])
    
    # Pre-commit hooks (optionnel)
    try:
        subprocess.run([pip_cmd, "install", "pre-commit"])
        subprocess.run(["pre-commit", "install"])
        click.echo("Pre-commit hooks install�s")
    except:
        click.echo("Pre-commit hooks non install�s (optionnel)")
    
    click.echo("Setup termin� !")

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
    click.echo("D�marrage de l'agent...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").absolute())
    subprocess.run([sys.executable, "src/agents/agentic_rag.py"], env=env)

@cli.command()
def serve():
    """Lancer l'API FastAPI"""
    click.echo("D�marrage du serveur API...")
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "src.api.main:app", 
        "--reload", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ])

@cli.command()
def notebook():
    """Lancer Jupyter pour exp�rimentation"""
    click.echo("D�marrage de Jupyter...")
    subprocess.run([sys.executable, "-m", "jupyter", "lab", "notebooks/"])

@cli.command()
def ollama_setup():
    """V�rifier et configurer Ollama"""
    click.echo("Configuration d'Ollama...")
    
    # V�rifier si Ollama est install�
    try:
        result = subprocess.run(["ollama", "version"], capture_output=True, text=True)
        click.echo(f"Ollama install� : {result.stdout.strip()}")
    except FileNotFoundError:
        click.echo("Ollama non trouv�. Installez-le depuis https://ollama.ai")
        return
    
    # T�l�charger les mod�les n�cessaires
    models = ["llama2", "nomic-embed-text"]
    for model in models:
        click.echo(f"T�l�chargement de {model}...")
        subprocess.run(["ollama", "pull", model])
    
    click.echo("Ollama configur� avec succ�s!")

@cli.command()
def clean():
    """Nettoyer les fichiers temporaires"""
    click.echo("Nettoyage...")
    
    import shutil
    
    # Dossiers � nettoyer
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
                click.echo(f"Supprim� : {path}")
            elif path.is_file():
                path.unlink()
                click.echo(f"Supprim� : {path}")
    
    click.echo("Nettoyage termin�!")

if __name__ == "__main__":
    cli()
