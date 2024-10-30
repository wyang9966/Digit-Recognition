import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

def load_model():
    try:
        model = tf.keras.models.load_model('digit_recognition_model.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print("Error loading model. Make sure 'digit_recognition_model.h5' is available.")
        raise e
    return model


class DrawPad:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Digit Recognition Draw Pad")

        # Create a canvas for drawing
        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()

        # Create a clear button
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        # Create a predict button
        self.predict_button = tk.Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        # Initialize PIL image to draw on
        self.image = Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)

    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        r = 8  # Radius of the circle
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', width=0)
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        # Resize and invert image
        img = self.image.resize((28, 28)).convert('L')
        img = np.array(img)
        img = 255 - img  # Invert image colors
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Predict the digit
        prediction = self.model.predict(img)
        digit = np.argmax(prediction)

        # Display result
        result_window = tk.Toplevel(self.root)
        result_window.title("Prediction Result")
        result_label = tk.Label(result_window, text=f"Predicted Digit: {digit}", font=("Arial", 20))
        result_label.pack()


# Initialize the GUI
if __name__ == "__main__":
    model = load_model()
    root = tk.Tk()
    app = DrawPad(root, model)
    root.mainloop()
