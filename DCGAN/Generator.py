from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load your trained generator model
generator_path = '/content/drive/MyDrive/model/malignant/generator_model_epoch_50.h5'  # Replace with the correct path
generator = load_model(generator_path)

# Function to generate, display, and save images
def generate_display_save_images(num_images=10, save_path='/content/drive/MyDrive/Generated_images/malignant'):
    latent_dim = 128 

    noise = np.random.randn(num_images, latent_dim)
    generated_images = generator.predict(noise)

    # Denormalize
    generated_images = (generated_images + 1) * 127.5
    generated_images = generated_images.astype(np.uint8)

    # Display and save images
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(generated_images[i])
        plt.axis('on')
        plt.title(f"Generated Image {i+1}\n(Malignant)")
        plt.imsave(os.path.join(save_path, f"generated_image_{i+1}.png"), generated_images[i])

    plt.tight_layout()
    plt.show()

# Generate, display, and save example images
generate_display_save_images(num_images=5)