from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras import layers, preprocessing
import numpy as np

data_dir = '/content/drive/MyDrive/preprocessed/malignant'
img_height = 224  
img_width = 224  
batch_size = 32
train_test_split = 0.8

image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=42
)

total_samples = len(image_dataset)
train_size = int(train_test_split * total_samples)
test_size = total_samples - train_size
train_ds = image_dataset.take(train_size)
test_ds = image_dataset.skip(train_size)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x: normalization_layer(x))
test_ds = test_ds.map(lambda x: normalization_layer(x))

latent_dim = 128

def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*1024, input_dim=latent_dim))
    model.add(layers.Reshape((7, 7, 1024)))
    model.add(layers.Conv2DTranspose(512, (5, 5), strides=2, padding='same'))  # Upsample to 14x14
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=2, padding='same'))  # Upsample to 28x28
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=2, padding='same'))  # Upsample to 56x56
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same'))  # Upsample to 112x112
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=2, padding='same', activation='tanh'))  # Upsample to 224x224
    return model

generator = build_generator()

def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=2, padding='same', input_shape=(224, 224, 3)))  
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(128, (5, 5), strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(256, (5, 5), strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

discriminator = build_discriminator()

discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
discriminator.trainable = False

gan_input = layers.Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

epochs = 50

d_losses_real, d_accs_real, g_losses, d_losses_fake, d_accs_fake = [], [], [], [], []

def train():
    for epoch in range(epochs):
        d_loss_real_epoch, d_acc_real_epoch, g_loss_epoch, d_loss_fake_epoch, d_acc_fake_epoch = 0, 0, 0, 0, 0
        for batch, X_train in enumerate(train_ds):
            real_images = X_train  
            noise = np.random.randn(batch_size, latent_dim)
            fake_images = generator.predict(noise)
            d_loss_real, d_acc_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
            d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
            noise = np.random.randn(batch_size, latent_dim)
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
            d_loss_real_epoch += d_loss_real
            d_acc_real_epoch += d_acc_real
            g_loss_epoch += g_loss
            d_loss_fake_epoch += d_loss_fake
            d_acc_fake_epoch += d_acc_fake
            print(f"Epoch: {epoch+1}, Batch: {batch}, [D loss: {d_loss_real:.4f}, acc: {d_acc_real:.2%}] [G loss: {g_loss:.4f}]")
        d_losses_real.append(d_loss_real_epoch / len(train_ds))
        d_accs_real.append(d_acc_real_epoch / len(train_ds))
        g_losses.append(g_loss_epoch / len(train_ds))
        d_losses_fake.append(d_loss_fake_epoch / len(train_ds))
        d_accs_fake.append(d_acc_fake_epoch / len(train_ds))
        if (epoch + 1) % 2 == 0:
            generator.save(f"/content/drive/MyDrive/model/malignant/generator_model_epoch_{epoch+1}.h5")
            discriminator.save(f"/content/drive/MyDrive/model/malignant/discriminator_model_epoch_{epoch+1}.h5")

train()

# Plotting accuracies
import matplotlib.pyplot as plt

epochs_range = range(1, epochs+1)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, d_accs_real, label='Discriminator Real Accuracy')
plt.plot(epochs_range, d_accs_fake, label='Discriminator Fake Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Discriminator Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, g_losses, label='Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Generator Loss')
plt.legend()

plt.tight_layout()
plt.show()