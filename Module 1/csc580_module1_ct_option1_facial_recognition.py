"""
  Kiran Ponappan Sreekumari 
  CSC580 – Applying Machine Learning and Neural Networks - Capstone
  Colorado State University - Global
  Dr. Pubali Banerjee 
  April 18, 2024
  Module 1: Critical Thinking Assignment - Option #1: Building a Basic Facial Recognition Program
  Write Python code to detect faces in an image.
"""
import face_recognition
from PIL import Image, ImageDraw
from os.path import dirname, join
import matplotlib.pyplot as plt 
import matplotlib.image as pltimg 

current_dir = dirname(__file__)
image_path = join(current_dir, "./image.jpg")

# Load the image file
image = face_recognition.load_image_file(image_path)

# Find all face locations in the image
face_locations = face_recognition.face_locations(image)

# Load the image into a PIL Image object so that we can draw on top of it
pil_image = Image.fromarray(image)

# Create a PIL drawing object
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the image
for face_location in face_locations:
    # Extract the coordinates of the face
    top, right, bottom, left = face_location

    # Draw a box around the face
    draw.rectangle([left, top, right, bottom], outline="red", width=7)

# Display the image with the face(s) identified
pil_image.save(join(current_dir, "./face_recognition.jpg"))

source_img = pltimg.imread(join(current_dir, "./image.jpg"))
processed_img = pltimg.imread(join(current_dir, "./face_recognition.jpg"))

##Plot all images
fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].imshow(source_img)
ax[0].set_title("Original Image")
ax[1].imshow(processed_img)
ax[1].set_title("Face Recognition Image")
plt.show()
