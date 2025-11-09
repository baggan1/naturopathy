#!/bin/bash
set -e

echo "🔹 Initializing Git LFS and preparing to push model folder..."

# 1️⃣ Ensure git and git-lfs are installed
if ! command -v git &> /dev/null
then
    echo "❌ Git not found. Please install Git before continuing."
    exit 1
fi

if ! command -v git-lfs &> /dev/null
then
    echo "📦 Installing Git LFS..."
    git lfs install
else
    echo "✅ Git LFS already installed."
fi

# 2️⃣ Initialize repository if not already done
if [ ! -d .git ]; then
    echo "📁 No Git repo found — initializing new one..."
    git init
    git remote add origin YOUR_GITHUB_REPO_URL
fi

# 3️⃣ Track all model files via Git LFS
git lfs install
git lfs track "model/**"

# 4️⃣ Stage and commit
git add .gitattributes
git add model
git commit -m "Add local SentenceTransformer model via Git LFS"

# 5️⃣ Push to GitHub
echo "🚀 Pushing model folder to GitHub using LFS..."
git push origin main

echo "✅ Done! Your model folder is now tracked with Git LFS and will deploy correctly on Render."
