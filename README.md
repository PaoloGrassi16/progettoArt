# MusicalAI

Framework neurale per analisi estetica e generazione procedurale di soundscapes,
basato sul progetto "MusicalAI: Un Framework Neurale dalla Percezione Estetica
alla Generazione Procedurale di Soundscapes".

Questo repository implementa:

- estrazione di spettrogrammi Log-Mel;
- SpecAugment con frequency masking e time masking;
- classificatore CNN-BiGRU con Self-Attention e Focal Loss;
- DCGAN per sintesi di spettrogrammi;
- validazione closed-loop tramite classificatore percettivo congelato;
- ricostruzione audio con Griffin-Lim;
- tre scenari reali: Gaming/VR, Pro-Audio e Musicoterapia Ambientale.



## Dataset

Il progetto e' pensato per dataset organizzati per cartelle di genere; nel nostro caso è stato utilizzato
il dataset GTZAN:

```text
data/genres_original/
  blues/*.wav
  classical/*.wav
  country/*.wav
  disco/*.wav
  hiphop/*.wav
  jazz/*.wav
  metal/*.wav
  pop/*.wav
  reggae/*.wav
  rock/*.wav
```

## Esecuzione rapida

Generazione di feature Log-Mel:

```bash
python -m musicalai.cli prepare-data --audio-root data/genres_original --output data/features
```

Training classificatore percettivo:

```bash
python -m musicalai.cli train-classifier --feature-root data/features --output runs/perception
```

Training DCGAN:

```bash
python -m musicalai.cli train-gan --feature-root data/features --output runs/dcgan
```

Generazione closed-loop:

```bash
python -m musicalai.cli generate --target ambient --output outputs/generated
```

Demo dei tre scenari:

```bash
python -m musicalai.scenarios.gaming_vr --intensity 0.8 --target electronic
python -m musicalai.scenarios.pro_audio --style lofi_glitch --target hiphop
python -m musicalai.scenarios.therapy_masking --noise-level 0.65 --target ambient
```

Gli output JSON di esempio sono in `examples/scenario_outputs/`.


## Struttura

```text
musicalai/
  audio/          Feature extraction, Log-Mel, inversione Griffin-Lim
  data/           Dataset e salvataggio feature
  models/         CNN-BiGRU-Attention, DCGAN, loss
  generation/     Closed-loop validator
  scenarios/      Applicazioni reali
  cli.py          Interfaccia a riga di comando
docs/             Guida GitHub, PDF e scenari reali
examples/         Output dimostrativi dei tre scenari
tests/            Test leggeri sulle parti pure Python/NumPy
```

## Nota sui checkpoint

I comandi funzionano con checkpoint reali se addestrati localmente. Per una
presentazione didattica, gli scenari possono essere eseguiti anche in modalita'
dry-run per mostrare la logica decisionale senza caricare pesi neurali.
