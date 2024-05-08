# Directory containing RGB images
input_dir = r"/content/drive/MyDrive/rgb_data/test/malignant"

# Directory to save images with Gaussian filter applied
output_dir = r"/content/drive/MyDrive/gaussian_data/test/malignant"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all files in the input directory
rgb_files = os.listdir(input_dir)

# Loop through each file in the input directory
for filename in rgb_files:
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # Read the RGB image
        rgb_image = cv2.imread(os.path.join(input_dir, filename))

        # Apply Gaussian filter
        gaussian_image = cv2.GaussianBlur(rgb_image, (5, 5), 0)

        # Save the image with Gaussian filter applied
        cv2.imwrite(os.path.join(output_dir, filename), gaussian_image)

print("Gaussian filter applied and images saved.")