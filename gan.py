import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

import decoder, discriminator, encoder, vae

vae_decoder = decoder.Decoder()
gan_discriminator = discriminator.Discriminator()

# Generate random noise (latent vector) as input for the VAE decoder
def generate_fake_images(batch_size, latent_dim=2):
    noise = tf.random.normal(shape=(batch_size, latent_dim))
    fake_images = vae_decoder(noise)
    return fake_images

# Function to visualize generated images
def plot_generated_images(fake_images, num_images=10):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(fake_images[i].numpy().reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()

fake_images = generate_fake_images(batch_size=25, latent_dim=2)

plot_generated_images(fake_images, num_images=10)
