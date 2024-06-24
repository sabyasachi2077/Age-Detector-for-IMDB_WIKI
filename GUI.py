import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model('final_model.h5')

# Function to preprocess the image and predict age
def predict_age(image_path):
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    age = model.predict(image)
    return age[0][0]

# Function to load and display the image
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        age = predict_age(file_path)
        result_label.config(text=f"Predicted Age: {int(age)}")

# Create the GUI window
root = tk.Tk()
root.title("Age Prediction")

# Add a button to load image
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack()

# Add a panel to display the image
panel = tk.Label(root)
panel.pack()

# Add a label to display the result
result_label = tk.Label(root, text="Predicted Age: ")
result_label.pack()

# Start the GUI event loop
root.mainloop()


