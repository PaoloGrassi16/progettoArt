# Applicazione ai tre scenari reali

## 1. Soundtrack adattiva per Gaming/VR

Comando:

```bash
python -m musicalai.cli scenario-gaming-vr --intensity 0.8 --target electronic
```

Il sistema traduce lo stato interattivo in parametri musicali: densita' ritmica,
BPM suggerito, durata del loop e soglia closed-loop. Il target `electronic`
viene mappato su `disco` quando si usa GTZAN, perche' e' la classe piu' vicina
nel benchmark.

## 2. Co-pilota creativo Pro-Audio

Comando:

```bash
python -m musicalai.cli scenario-pro-audio --style lofi_glitch --target hiphop
```

Lo scenario produce richieste per sample e texture sonore inedite. I controlli
`glitch_amount`, `warmth` e `novelty` guidano la generazione e rendono esplicito
il legame con l'estetica Lo-Fi/Glitch Art descritta nel progetto.

## 3. Musicoterapia e mascheramento acustico

Comando:

```bash
python -m musicalai.cli scenario-therapy --noise-level 0.65 --target ambient
```

Lo scenario trasforma il livello di rumore in forza di mascheramento, riduzione
dei transienti e ammorbidimento delle basse frequenze. Il target `ambient`
viene mappato su `classical` nel caso GTZAN.

## Integrazione con la DCGAN closed-loop

Dopo il training, ogni piano scenario puo' essere trasformato in audio con:

```bash
python -m musicalai.cli generate ^
  --target disco ^
  --classifier-checkpoint runs/perception/perception_classifier.pt ^
  --generator-checkpoint runs/dcgan/dcgan_generator.pt ^
  --output outputs/generated ^
  --threshold 0.85
```

Il classificatore CNN-BiGRU-Attention congelato accetta solo candidati con
confidenza sufficiente, riproducendo il gating descritto nella sezione
Closed-Loop DCGAN del PDF.

