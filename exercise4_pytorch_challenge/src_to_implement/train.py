import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv('data.csv', sep=';')
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=None)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataset = ChallengeDataset(train_data, 'train')
val_dataset = ChallengeDataset(val_data, 'val')

train_loader = t.utils.data.DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4 if t.cuda.is_available() else 0
)
val_loader = t.utils.data.DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=4 if t.cuda.is_available() else 0
)

# create an instance of our ResNet model
resnet_model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion

# For multi-label classification where classes are NOT mutually exclusive:
# - Our model already includes sigmoid activation (outputs in [0,1])
# - Use BCELoss which expects sigmoid-activated probabilities
# - DO NOT use BCEWithLogitsLoss (that would apply sigmoid twice!)
# - DO NOT use CrossEntropyLoss (assumes mutually exclusive classes)
criterion = t.nn.BCELoss()

# Set up optimizer - Adam is a good choice for deep learning
optimizer = t.optim.Adam(resnet_model.parameters(), lr=0.001, weight_decay=1e-4)

# Create checkpoints directory if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)

# Create trainer with early stopping
trainer = Trainer(
    model=resnet_model,
    crit=criterion,
    optim=optimizer,
    train_dl=train_loader,
    val_test_dl=val_loader,
    cuda=t.cuda.is_available(),
    early_stopping_patience=15  # More patience for first training
)

print("Starting training with these hyperparameters:")
print(f"Learning Rate: 0.001")
print(f"Batch Size: 32")
print(f"Validation Split: 15%")
print(f"Weight Decay: 1e-4")
print(f"Early Stopping Patience: 15 epochs")
print(f"Using device: {'CUDA' if t.cuda.is_available() else 'CPU'}")

# go, go, go... call fit on trainer
res = trainer.fit(epochs=50)

# plot the results
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(res[0])), res[0], label='train loss', color='blue')
plt.plot(np.arange(len(res[1])), res[1], label='val loss', color='red')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('losses.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nTraining completed!")
print(f"Final training loss: {res[0][-1]:.4f}")
print(f"Final validation loss: {res[1][-1]:.4f}")
print(f"Best validation loss: {min(res[1]):.4f}")
print(f"Total epochs: {len(res[0])}")