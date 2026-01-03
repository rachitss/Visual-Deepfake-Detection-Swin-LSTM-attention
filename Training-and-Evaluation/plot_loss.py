import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss_path):
    training_loss = np.load(loss_path)
    epochs = list(range(1, len(training_loss) + 1))
    plt.figure(figsize=(16, 8))
    plt.plot(epochs, training_loss)
    plt.title('Training Log Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Log Loss')
    plt.show()

    print(f"Last Epoch Training Log Loss: {training_loss[-1]:.4f}")
    print(f"Number of epochs trained: {len(training_loss)}")

if __name__ == '__main__':
    loss_path = '<path-to-loss-file>'
    plot_loss(loss_path)
