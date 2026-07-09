# Inserimento del link GitHub nel PDF

Dopo aver pubblicato la repository, aggiungi il link al PDF originale con:

```bash
python scripts/add_github_link_to_pdf.py ^
  --input "C:\Users\furtr\Downloads\progettoArt (1).pdf" ^
  --output "C:\Users\furtr\Documents\Codex\2026-07-09\de\outputs\progettoArt_with_github.pdf" ^
  --github-url "https://github.com/PaoloGrassi16/progettoArt.git"
```

Su macOS/Linux sostituisci `^` con `\`.

Il link viene stampato nella prima pagina e reso cliccabile.
