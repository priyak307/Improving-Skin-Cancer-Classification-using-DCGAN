from google.colab import drive
drive.mount('/content/drive')

import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/archive.zip', 'r') as zip_ref:
    zip_ref.extractall('/content')

from PIL import Image
import os

# Directory containing GBR images
input_dir = "/content/data/test/malignant"

# Directory to save RGB images
output_dir = "/content/drive/MyDrive/rgb_data/test/malignant"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all files in the input directory
gbr_files = os.listdir(input_dir)

# Loop through each file in the input directory
for filename in gbr_files:
    # Check if the file is an image
    if filename.endswith(".jpg"):
        # Open the GBR image
        gbr_image = Image.open(os.path.join(input_dir, filename))

        # Convert to RGB
        rgb_image = gbr_image.convert("RGB")

        # Save the RGB image
        rgb_image.save(os.path.join(output_dir, filename.replace(".jpg", ".jpg")))

print("Conversion complete.")