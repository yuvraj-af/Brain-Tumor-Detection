import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
from keras.models import load_model

model = load_model('brain_tumor_detection_model.h5')

img_width, img_height = 64, 64

image_path = ""

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  
    return img

def classify_image():
    if not image_path:
        result_label.config(text='No Image Selected')
        return

    image = preprocess_image(image_path)
    prediction = model.predict(np.array([image]))
    result = 'Tumor Detected' if prediction[0][0] > 0.5 else 'No Tumor Detected'
    result_label.config(text=result)

def choose_image():
    global image_path
    image_path = filedialog.askopenfilename(title='Choose Image', filetypes=[('Image Files', '*.jpg;*.jpeg;*.png')])
    if image_path:
        result_label.config(text='')

        img = Image.open(image_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

window = tk.Tk()
window.title('Brain Tumor Detection App')
window.geometry('600x500')

button_frame = tk.Frame(window)
button_frame.pack(side='bottom', pady=(0, 50))

choose_image_btn = tk.Button(button_frame, text='Choose Image', command=choose_image, padx=10, pady=5, font=('Arial', 16), bg='light grey')
choose_image_btn.grid(row=0, column=0, padx=50)

test_btn = tk.Button(button_frame, text='Test', command=classify_image, padx=10, pady=5, font=('Arial', 16), bg='light grey')
test_btn.grid(row=0, column=1, padx=50)

button_frame.grid_rowconfigure(0, weight=1)
button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)

image_label = tk.Label(window)
image_label.pack(pady=10)

result_label = tk.Label(window, text='', font=('Arial', 22))
result_label.pack(pady=10)

welcome_label = tk.Label(window, text='Select an Image', font=('Arial', 24))
welcome_label.pack(pady=20)

window.mainloop()