#!/bin/bash
set -e  # Exit on error

echo "Saving all changes to GitHub..."

# Stage, commit, and push
git add .
git config --global user.email "user@example.com"
git config --global user.name "GitHub User"
git commit -m "Save project snapshot" || echo "No changes to commit."
git push origin main

# Print the shareable link
echo
echo "✅ SUCCESS! Share this link with your selector:"
echo "https://github.com/$GITHUB_REPOSITORY"
echo
