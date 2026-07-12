import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('runs/perception/classifier_history.csv')

plt.figure(figsize=(12, 5))

# Sotto-grafico 1: Loss
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='firebrick', marker='o')
plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='navy', linestyle='--', marker='s')
plt.title('Andamento della Loss')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.grid(True, linestyle=':')
plt.legend()

# Sotto-grafico 2: Accuracy
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['train_acc'], label='Train Acc', color='darkgreen', marker='o')
plt.plot(df['epoch'], df['val_acc'], label='Val Acc', color='darkorange', linestyle='--', marker='s')
plt.title('Andamento dell\'Accuracy')
plt.xlabel('Epoche')
plt.ylabel('Accuracy (0.0 - 1.0)')
plt.grid(True, linestyle=':')
plt.legend()

plt.tight_layout()
plt.savefig(
    'curve_apprendimento_classificatore.svg',
    bbox_inches='tight'
)
plt.show()