# Publish To GitHub

This folder is prepared as a standalone GitHub-ready repository.

Suggested steps:

1. Create an empty GitHub repository.
2. Open a terminal in this folder.
3. Run:

```powershell
git init -b main
git add .
git commit -m "Initial public release of the observation-layer framework"
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

Notes:

- model weights are intentionally excluded
- private or heavy source videos are excluded
- tracking frame dumps are excluded
- benchmark summaries, scripts, notes, prompts, and compact result assets are included
