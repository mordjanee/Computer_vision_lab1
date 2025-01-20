import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import requests
import encoder, decoder, vae

# Load MNIST dataset
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
response = requests.get(url)
with open("mnist.npz", "wb") as f:
    f.write(response.content)

# Load the dataset
data = np.load("mnist.npz")
x_train, x_test = data["x_train"], data["x_test"]

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

latent_dim = 2  # Latent space dimension


# Initialize encoder, decoder, and VAE
encoder_model = encoder.Encoder(latent_dim)
decoder_model = decoder.Decoder()
vae_model = vae.VAE(encoder_model, decoder_model)

# Loss Function
def vae_loss(x, reconstruction, mean, log_var):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, reconstruction))
    reconstruction_loss *= 28 * 28
    kl_divergence = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var))
    return reconstruction_loss + kl_divergence

# Optimizer
optimizer = tf.keras.optimizers.Adam()

# Training Loop
@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        reconstruction, mean, log_var = vae_model(x)
        loss = vae_loss(x, reconstruction, mean, log_var)
    gradients = tape.gradient(loss, vae_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae_model.trainable_variables))
    return loss

# Training process
epochs = 20
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)

for epoch in range(epochs):
    for step, x_batch in enumerate(train_dataset):
        loss = train_step(x_batch)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")


# Visualiser les images reconstruites après l'entraînement
def plot_reconstructed_images(model, x_test):
    reconstruction, _, _ = model(x_test)
    plt.figure(figsize=(10, 10))
    for i in range(10):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(reconstruction[i].numpy().reshape(28, 28))
        plt.axis("off")
    plt.show()

# Appeler cette fonction après l'entraînement
plot_reconstructed_images(vae_model, x_test[:10])

