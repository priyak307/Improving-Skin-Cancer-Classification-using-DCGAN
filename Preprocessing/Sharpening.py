from PIL import Image, ImageFilter

# Directory containing RGB images
input_dir = r"/content/drive/MyDrive/gaussian_data/test/malignant"

# Directory to save sharpened images
output_dir = r"/content/drive/MyDrive/sharpening_data/test/malignant"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all files in the input directory
rgb_files = os.listdir(input_dir)

# Loop through each file in the input directory
for filename in rgb_files:
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # Open the RGB image
        rgb_image = Image.open(os.path.join(input_dir, filename))

        # Apply sharpening filter
        sharpened_image = rgb_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

        # Save the sharpened image
        sharpened_image.save(os.path.join(output_dir, filename))

print("Sharpening complete.")