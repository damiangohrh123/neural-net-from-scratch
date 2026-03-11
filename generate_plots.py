import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Data from terminal logs
epochs = [1, 2, 3, 4, 5]
loss = [0.2320, 0.1028, 0.0726, 0.0550, 0.0424]
test_accuracy = [96.50, 96.72, 97.23, 97.58, 97.82]
train_accuracy = [97.40, 98.15, 98.80, 99.10, 99.45] 

# Plot 1: Convergence (Loss vs. Test Accuracy) 
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Avg Training Loss', color='tab:blue', fontsize=12)
ax1.plot(epochs, loss, marker='o', color='tab:blue', linewidth=2, label='Training Loss')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, linestyle='--', alpha=0.4)
ax2 = ax1.twinx()
ax2.set_ylabel('Test Accuracy (%)', color='tab:orange', fontsize=12)
ax2.plot(epochs, test_accuracy, marker='s', color='tab:orange', linewidth=2, label='Test Accuracy')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.title('Training Convergence', fontsize=14, pad=15)
fig.tight_layout()
plt.savefig('./images/loss_vs_accuracy.png', dpi=300)

# Plot 2: Overfitting Check (Train Accuracy vs. Test Accuracy)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, marker='o', color='tab:blue', linewidth=2, label='Training Accuracy')
plt.plot(epochs, test_accuracy, marker='s', color='tab:orange', linewidth=2, label='Test Accuracy')
plt.title('Overfitting Analysis', fontsize=14, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(90, 100)
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('./images/overfitting_check.png', dpi=300)
plt.show()