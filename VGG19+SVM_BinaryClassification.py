from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model

# Step 1: Load and preprocess the dataset
data_dir = "/content/drive/MyDrive/preprocessed" 
benign_dir = os.path.join(data_dir, "/content/drive/MyDrive/preprocessed/benign")
malignant_dir = os.path.join(data_dir, "/content/drive/MyDrive/preprocessed/malignant")
# Function to load images from a directory
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        img = cv2.resize(img, (224, 224))
        images.append(img)
    return np.array(images)

benign_images = load_images(benign_dir)
malignant_images = load_images(malignant_dir)

# Step 2: Visualize images
def visualize_images(images, title):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

visualize_images(benign_images, "Benign Images")
visualize_images(malignant_images, "Malignant Images")

# Step 3: Prepare data for training
X = np.concatenate((benign_images, malignant_images), axis=0)
y = np.concatenate((np.zeros(len(benign_images)), np.ones(len(malignant_images))), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build VGG-19 + SVM Architecture
vgg_model = VGG19(include_top=False, input_shape=(224, 224, 3))
x = vgg_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=vgg_model.input, outputs=predictions)
svm_model = SVC(kernel='linear', C=1.0)

model.summary()

# Step 5: Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# Step 6: Plot training and validation loss and accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

from sklearn.metrics import roc_curve, roc_auc_score

# Step 7: After training the model and obtaining predictions, calculate ROC curve and AUC score
y_pred = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred)
print(f"AUC Score: {auc_score:.4f}")