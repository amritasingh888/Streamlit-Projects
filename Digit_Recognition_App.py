# Import Libraries
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Load and Train MNIST Model
# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0
# Build model
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dropout(0.3),
tf.keras.layers.Dense(10, activation='softmax')
])
# Compile model
model.compile(
optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)
# Train model
model.fit(x_train, y_train, epochs=3)
# Create User Interface
st.title("Digit Recognition App")
st.write("Draw a digit or upload an image to predict.")

# Drawing Canvas
canvas_result = st_canvas(
 fill_color="black",
 stroke_width=10,
 stroke_color="white",
 background_color="black",
 height=280,
 width=280,
 drawing_mode="freedraw",
 key="canvas"
)
# Upload Image Option
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
# Image Preprocessing
def preprocess_image(img):
    img = img.convert('L') # grayscale
    img = img.resize((28, 28)) # resize
    img = np.array(img)
    img = 255 - img # invert (white digit on black)
    img = img / 255.0 # normalize
    img = img.reshape(1, 28, 28, 1) # CNN shape
    return img
# Handle Input (Draw or Upload)
image = None
# From canvas
if canvas_result.image_data is not None:
    image = Image.fromarray(canvas_result.image_data.astype('uint8'))
# From upload
if uploaded_file is not None:
    image = Image.open(uploaded_file)
if image is not None:
    st.image(image, caption="Input Image", width=150)

# Prediction
if image is not None:
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_digit = np.argmax(prediction)
    st.subheader("Prediction")
    st.success(f"Predicted Digit: {predicted_digit}")