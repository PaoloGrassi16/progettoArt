# Pubblicazione su GitHub

## Opzione con GitHub CLI

Da questa cartella:

```bash
git init
git add .
git commit -m "Add MusicalAI project implementation"
gh repo create progettoArt --public --source . --remote origin --push
```

Il comando finale stampa l'URL della repository, ad esempio:

```text
https://github.com/PaoloGrassi16/progettoArt.git
```

## Opzione da browser

1. Crea una nuova repository su GitHub chiamata `progettoArt`.
2. Da terminale, dentro questa cartella:

```bash
git init
git add .
git commit -m "Add MusicalAI project implementation"
git branch -M main
git remote add origin https://github.com/PaoloGrassi16/progettoArt.git
git push -u origin main
```
