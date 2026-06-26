#!/usr/bin/env bash
set -e

# Create modular project structure.
mkdir -p agent
mkdir -p backend
mkdir -p ui
mkdir -p notebooks

# Create Python package markers.
touch agent/__init__.py
touch backend/__init__.py

# Create source files.
touch agent/core.py
touch agent/tools.py
touch agent/prompts.py
touch backend/api.py
touch ui/app.py

# Create notebook placeholder.
touch notebooks/.gitkeep

# Create config/docs files.
touch .gitignore
touch requirements.txt
touch .env.example
touch README.md

echo "Repository structure created successfully."
