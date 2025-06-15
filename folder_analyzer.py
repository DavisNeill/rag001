#!/usr/bin/env python3
"""
Script pour analyser la structure d'un dossier et générer un rapport
que vous pouvez copier-coller pour analyse par Claude
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class FolderAnalyzer:
    """Analyseur de structure de dossier"""
    
    def __init__(self, root_path: str = ".", max_depth: int = 3, max_files_per_dir: int = 20):
        self.root_path = Path(root_path).resolve()
        self.max_depth = max_depth
        self.max_files_per_dir = max_files_per_dir
        
        # Extensions importantes à analyser
        self.important_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.md', '.rst', '.txt', '.csv',
            '.sql', '.db', '.sqlite',
            '.html', '.css', '.scss',
            '.docker', '.dockerfile',
            '.gitignore', '.env', '.envrc'
        }
        
        # Dossiers à ignorer
        self.ignore_dirs = {
            '__pycache__', '.git', '.vscode', '.idea', 'node_modules',
            '.pytest_cache', '.coverage', 'htmlcov', 'dist', 'build',
            '.tox', '.mypy_cache', '.cache', 'venv', 'env', '.venv',
            '.DS_Store', 'Thumbs.db'
        }
    
    def analyze_structure(self) -> Dict[str, Any]:
        """Analyse la structure du dossier"""
        
        analysis = {
            "metadata": {
                "root_path": str(self.root_path),
                "analysis_time": datetime.now().isoformat(),
                "total_files": 0,
                "total_dirs": 0
            },
            "structure": {},
            "file_types": {},
            "important_files": [],
            "potential_issues": [],
            "summary": {}
        }
        
        # Analyser récursivement
        self._analyze_directory(
            self.root_path, 
            analysis["structure"], 
            analysis, 
            depth=0
        )
        
        # Générer le résumé
        self._generate_summary(analysis)
        
        return analysis
    
    def _analyze_directory(self, path: Path, structure: Dict, analysis: Dict, depth: int):
        """Analyse récursive d'un répertoire"""
        
        if depth > self.max_depth:
            return
        
        if path.name in self.ignore_dirs:
            return
        
        try:
            items = list(path.iterdir())
        except PermissionError:
            analysis["potential_issues"].append(f"Permission denied: {path}")
            return
        
        # Séparer dossiers et fichiers
        dirs = [item for item in items if item.is_dir() and item.name not in self.ignore_dirs]
        files = [item for item in items if item.is_file()]
        
        analysis["metadata"]["total_dirs"] += len(dirs)
        analysis["metadata"]["total_files"] += len(files)
        
        # Analyser les fichiers
        structure["files"] = []
        for file_path in files[:self.max_files_per_dir]:
            file_info = self._analyze_file(file_path, analysis)
            structure["files"].append(file_info)
        
        if len(files) > self.max_files_per_dir:
            structure["files"].append(f"... et {len(files) - self.max_files_per_dir} autres fichiers")
        
        # Analyser les sous-dossiers
        structure["directories"] = {}
        for dir_path in dirs:
            structure["directories"][dir_path.name] = {}
            self._analyze_directory(
                dir_path, 
                structure["directories"][dir_path.name], 
                analysis, 
                depth + 1
            )
    
    def _analyze_file(self, file_path: Path, analysis: Dict) -> Dict[str, Any]:
        """Analyse un fichier individuel"""
        
        file_info = {
            "name": file_path.name,
            "size": file_path.stat().st_size,
            "extension": file_path.suffix.lower()
        }
        
        # Compter les types de fichiers
        ext = file_path.suffix.lower()
        if ext in analysis["file_types"]:
            analysis["file_types"][ext] += 1
        else:
            analysis["file_types"][ext] = 1
        
        # Marquer les fichiers importants
        if (ext in self.important_extensions or 
            file_path.name.lower() in ['readme.md', 'requirements.txt', 'package.json', 'dockerfile', 'makefile']):
            
            file_info["important"] = True
            analysis["important_files"].append(str(file_path.relative_to(self.root_path)))
            
            # Analyser le contenu des fichiers texte importants
            if ext in ['.py', '.js', '.json', '.md', '.txt', '.yml', '.yaml'] and file_path.stat().st_size < 10000:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    file_info["preview"] = content[:200] + "..." if len(content) > 200 else content
                    file_info["lines"] = len(content.splitlines())
                except Exception:
                    file_info["preview"] = "[Impossible de lire le fichier]"
        
        return file_info
    
    def _generate_summary(self, analysis: Dict):
        """Génère un résumé de l'analyse"""
        
        summary = analysis["summary"]
        
        # Statistiques générales
        summary["total_files"] = analysis["metadata"]["total_files"]
        summary["total_directories"] = analysis["metadata"]["total_dirs"]
        
        # Types de fichiers les plus courants
        sorted_types = sorted(analysis["file_types"].items(), key=lambda x: x[1], reverse=True)
        summary["top_file_types"] = sorted_types[:10]
        
        # Détection du type de projet
        summary["project_type"] = self._detect_project_type(analysis)
        
        # Fichiers de configuration détectés
        config_files = [f for f in analysis["important_files"] 
                       if any(config in f.lower() for config in ['requirements', 'package.json', 'dockerfile', 'makefile', '.env'])]
        summary["config_files"] = config_files
        
        # Recommandations
        summary["recommendations"] = self._generate_recommendations(analysis)
    
    def _detect_project_type(self, analysis: Dict) -> List[str]:
        """Détecte le type de projet basé sur les fichiers"""
        
        types = []
        important_files = [f.lower() for f in analysis["important_files"]]
        
        if any('requirements.txt' in f or 'setup.py' in f or '.py' in f for f in important_files):
            types.append("Python")
        
        if any('package.json' in f or '.js' in f or '.ts' in f for f in important_files):
            types.append("JavaScript/TypeScript")
        
        if any('dockerfile' in f for f in important_files):
            types.append("Docker")
        
        if any('.java' in f for f in important_files):
            types.append("Java")
        
        if any('readme' in f for f in important_files):
            types.append("Documented")
        
        return types
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Génère des recommandations d'amélioration"""
        
        recommendations = []
        important_files = [f.lower() for f in analysis["important_files"]]
        
        if not any('readme' in f for f in important_files):
            recommendations.append("Ajouter un README.md pour documenter le projet")
        
        if not any('.gitignore' in f for f in important_files):
            recommendations.append("Ajouter un .gitignore pour exclure les fichiers temporaires")
        
        if any('requirements.txt' in f for f in important_files) and not any('.env' in f for f in important_files):
            recommendations.append("Considérer l'ajout d'un fichier .env pour la configuration")
        
        if analysis["metadata"]["total_files"] > 100 and not any('test' in f for f in important_files):
            recommendations.append("Ajouter des tests unitaires (dossier tests/)")
        
        return recommendations
    
    def generate_report(self) -> str:
        """Génère un rapport textuel pour Claude"""
        
        analysis = self.analyze_structure()
        
        report = f"""# 📁 Analyse de Structure de Dossier

## 📊 Résumé Exécutif
- **Dossier analysé :** {analysis['metadata']['root_path']}
- **Total fichiers :** {analysis['summary']['total_files']}
- **Total dossiers :** {analysis['summary']['total_directories']}
- **Type de projet détecté :** {', '.join(analysis['summary']['project_type']) if analysis['summary']['project_type'] else 'Non déterminé'}

## 🗂️ Types de Fichiers Principaux
"""
        
        for ext, count in analysis['summary']['top_file_types']:
            ext_name = ext if ext else 'sans extension'
            report += f"- **{ext_name}** : {count} fichiers\n"
        
        report += f"\n## 📋 Fichiers Importants Détectés\n"
        for file_path in analysis['important_files'][:20]:  # Limiter à 20
            report += f"- {file_path}\n"
        
        if len(analysis['important_files']) > 20:
            report += f"- ... et {len(analysis['important_files']) - 20} autres fichiers importants\n"
        
        report += f"\n## 🌳 Structure des Dossiers\n"
        report += self._format_structure(analysis['structure'], prefix="")
        
        if analysis['summary']['config_files']:
            report += f"\n## ⚙️ Fichiers de Configuration\n"
            for config in analysis['summary']['config_files']:
                report += f"- {config}\n"
        
        if analysis['summary']['recommendations']:
            report += f"\n## 💡 Recommandations\n"
            for rec in analysis['summary']['recommendations']:
                report += f"- {rec}\n"
        
        if analysis['potential_issues']:
            report += f"\n## ⚠️ Problèmes Potentiels\n"
            for issue in analysis['potential_issues']:
                report += f"- {issue}\n"
        
        report += f"\n## 🔍 Aperçu des Fichiers Clés\n"
        
        # Montrer le contenu des fichiers les plus importants
        for file_info in self._get_key_files_content(analysis):
            report += f"\n### 📄 {file_info['path']}\n"
            if 'preview' in file_info:
                report += f"```\n{file_info['preview']}\n```\n"
        
        return report
    
    def _format_structure(self, structure: Dict, prefix: str = "", max_depth: int = 3) -> str:
        """Formate la structure en arbre textuel"""
        
        if not structure or max_depth <= 0:
            return ""
        
        result = ""
        
        # Afficher les fichiers
        if "files" in structure:
            for file_item in structure["files"]:
                if isinstance(file_item, dict):
                    marker = "📄" if not file_item.get("important") else "⭐"
                    result += f"{prefix}├── {marker} {file_item['name']}\n"
                else:  # String pour "... et X autres fichiers"
                    result += f"{prefix}├── 📁 {file_item}\n"
        
        # Afficher les dossiers
        if "directories" in structure:
            dirs = list(structure["directories"].items())
            for i, (dir_name, dir_content) in enumerate(dirs):
                is_last = (i == len(dirs) - 1)
                connector = "└──" if is_last else "├──"
                result += f"{prefix}{connector} 📂 {dir_name}/\n"
                
                # Récursion pour le contenu du dossier
                extension = "    " if is_last else "│   "
                result += self._format_structure(
                    dir_content, 
                    prefix + extension, 
                    max_depth - 1
                )
        
        return result
    
    def _get_key_files_content(self, analysis: Dict) -> List[Dict]:
        """Récupère le contenu des fichiers clés"""
        
        key_files = []
        priority_files = ['readme.md', 'requirements.txt', 'package.json', 'main.py', '__init__.py']
        
        for file_path in analysis['important_files'][:5]:  # Top 5 fichiers
            if any(priority in file_path.lower() for priority in priority_files):
                full_path = self.root_path / file_path
                if full_path.exists() and full_path.stat().st_size < 5000:
                    try:
                        content = full_path.read_text(encoding='utf-8', errors='ignore')
                        key_files.append({
                            'path': file_path,
                            'preview': content[:500] + "..." if len(content) > 500 else content
                        })
                    except Exception:
                        pass
        
        return key_files

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyser la structure d\'un dossier')
    parser.add_argument('path', nargs='?', default='.', help='Chemin du dossier à analyser')
    parser.add_argument('--depth', type=int, default=3, help='Profondeur maximum d\'analyse')
    parser.add_argument('--max-files', type=int, default=20, help='Nombre maximum de fichiers par dossier')
    parser.add_argument('--json', action='store_true', help='Sortie en format JSON')
    
    args = parser.parse_args()
    
    analyzer = FolderAnalyzer(args.path, args.depth, args.max_files)
    
    if args.json:
        analysis = analyzer.analyze_structure()
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    else:
        report = analyzer.generate_report()
        print(report)
        
        print("\n" + "="*60)
        print("📋 INSTRUCTIONS POUR CLAUDE :")
        print("Copiez le rapport ci-dessus et collez-le dans votre conversation")
        print("avec Claude pour obtenir une analyse détaillée de votre projet.")
        print("="*60)

if __name__ == "__main__":
    main()