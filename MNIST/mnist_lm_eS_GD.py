import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import HeNormal
import numpy as np
import os

# aggregator is a list of tensors
def compute_aggregator_norm(aggregator):
    squared_norms = [tf.reduce_sum(tf.square(g)) for g in aggregator]
    total_norm = tf.sqrt(tf.add_n(squared_norms))
    return total_norm

seed = 42
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu', kernel_initializer=HeNormal(seed=seed)),
    Dense(10, activation='softmax', kernel_initializer=HeNormal(seed=seed))
])

# 2. bis Build the pivot model
model_pivot = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu', kernel_initializer=HeNormal(seed=seed)),
    Dense(10, activation='softmax', kernel_initializer=HeNormal(seed=seed))
])

lr = 0.01
# 3. Loss and optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

# 4. Training parameters
epochs = 1000
epsilon = 1.e-4
batch_size = 12500  # Change this to len(x_train) for full batch
#batch_size = len(x_train) #for full batch

# Determine number of batches
num_batches = len(x_train) // batch_size+1
print(f"nb of batches = {num_batches:e}")

# Initialize an empty aggregator (list of same length as model.trainable_variables)
# Each aggregator is a list of tensors matching model.trainable_variables
aggregator     = [tf.zeros_like(var) for var in model.trainable_variables]
sum_aggregator = [tf.zeros_like(var) for var in model.trainable_variables]
model_pivot.set_weights(model.get_weights())

L = 0. 
batch_pivot = 0

# For tracking loss and accuracy
batch_losses = []
batch_accuracies = [] 


# 5. Training loop with batching
for epoch in range(epochs):
    # Shuffle the data at the start of each epoch
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    # Track epoch metrics
    epoch_loss = 0
    correct = 0
    total = 0

    #set sum of gradients to zero
    sum_aggregator = [tf.zeros_like(var) for var in model.trainable_variables]

    # going through the batches
    for batch_idx, i in enumerate(range(0, len(x_train), batch_size)):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        with tf.GradientTape(persistent=True) as tape:
            # Current weights forward pass
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        
            logits_pivot = model_pivot(x_batch, training=True)        
            loss_pivot = loss_fn(y_batch, logits_pivot)

        gradients       = tape.gradient(loss      , model.trainable_variables)
        gradients_pivot = tape.gradient(loss_pivot, model_pivot.trainable_variables)

        # remove the previously identified stiffest batch
        aggregator = [a - g for a, g in zip(aggregator, gradients_pivot)]
        # Sum gradients in sum_aggregator
        sum_aggregator = [a + g for a, g in zip(sum_aggregator, gradients)]
        # update aggregator
        aggregator = [a + g for a, g in zip(aggregator, gradients)]

        # Use aggregator to apply gradients
        optimizer.apply_gradients(zip(aggregator, model.trainable_variables))

        if batch_idx == num_batches-1:
            aggregator = sum_aggregator

        norm_current_grad = norm_current_grad = compute_aggregator_norm(gradients).numpy()
        if  norm_current_grad > L:
            L = norm_current_grad
            batch_pivot = batch_idx

        if batch_idx == batch_pivot:
            model_pivot.set_weights(model.get_weights())

        epoch_loss += loss.numpy() * len(x_batch)
        preds = tf.argmax(logits, axis=1)
        labels = tf.argmax(y_batch, axis=1)
        correct += tf.reduce_sum(tf.cast(preds == labels, tf.int32)).numpy()
        total += len(x_batch)

        # compute norm of aggregator
        norm = compute_aggregator_norm(aggregator)
    if norm < epsilon:
        print("norm aggregator < epsilon")
        break

    epoch_loss /= total
    accuracy = correct / total
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.4f}, Norm aggregator = {norm:e}")

    # Track per batch
    batch_losses.append(loss.numpy())
    batch_accuracy = tf.reduce_mean(tf.cast(preds == labels, tf.float32)).numpy()
    batch_accuracies.append(batch_accuracy)

# 6. Evaluate on test data
test_logits = model(x_test, training=False)
test_loss = loss_fn(y_test, test_logits).numpy()
test_preds = tf.argmax(test_logits, axis=1)
test_labels = tf.argmax(y_test, axis=1)
test_accuracy = tf.reduce_mean(tf.cast(test_preds == test_labels, tf.float32)).numpy()

print(f"\nTest Accuracy: {test_accuracy:.4f}")


import pandas as pd

# Save to CSV
metrics_df = pd.DataFrame({
    'batch_loss': batch_losses,
    'batch_accuracy': batch_accuracies
})
metrics_df.to_csv("SpethGD_Red_Mem_batch_metrics.csv", index_label="epochs")

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot Loss on the left y-axis
ax1.plot(batch_losses, label="Loss", color='tab:blue')
ax1.set_xlabel("epochs")
ax1.set_ylabel("Loss", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

# Create a second y-axis for accuracy
ax2 = ax1.twinx()
ax2.plot(batch_accuracies, label="Accuracy", color='tab:orange')
ax2.set_ylabel("Accuracy", color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Optional: Add aggregator norm as a third line (not a third axis)
# ax1.plot(aggregator_norms, label="Aggregator Norm", color='tab:green', linestyle='dashed')

# Title and layout
plt.title("Batch Loss and Accuracy")
fig.tight_layout()
fig.savefig("SpethGD_Red_Mem_loss_accuracy_plot.png", dpi=300)
plt.show()

