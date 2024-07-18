import cv2

# Replace 'path_to_image.png' with the actual path to your image file
image_path = 'images/img11.png'

# Read the image
img = cv2.imread(image_path)

# Check if the image was successfully loaded
if img is None:
    print(f"Failed to load image '{image_path}'")
else:
    # Get the dimensions of the image
    height, width, channels = img.shape

    print(f"Width: {width} pixels")
    print(f"Height: {height} pixels")
    print(f"Number of Channels: {channels}")
