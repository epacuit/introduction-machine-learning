import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Use a test MNIST image
(_, _), (test_images, _) = mnist.load_data()
test_image = test_images[0]
test_image_data = test_image.reshape((1, 28*28)).astype("float32") / 255


model = tf.keras.models.load_model('model.keras')
# Predict using the test image
prediction = model.predict(test_image_data)
st.write(prediction.argmax())  


# Load the trained Keras model
model = tf.keras.models.load_model('model.keras')

# Title and description
st.title("Example - Handwritten Digit Recognizer")
st.write("Draw a digit below, and the model will predict what number it is!")

# Create a canvas for user input
canvas_result = st_canvas(
    fill_color="#ffffff",  # Background color
    stroke_width=20,       # Brush thickness
    stroke_color="#000000",  # Brush color
    background_color="#ffffff",  # Canvas background color
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Process the drawn image if available
if canvas_result.image_data is not None:
    # Convert RGBA image to grayscale
    image = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    # Resize to 28x28 pixels
    resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA).reshape((1, 28*28)).astype("float32") / 255
    
    image = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)

    # Invert the image to match MNIST's black background and white digits
    inverted = cv2.bitwise_not(image)

    resized = cv2.resize(inverted, (28, 28), interpolation=cv2.INTER_AREA)
    # Resize to 28x28 pixels
    resized_data = resized.reshape((1, 28*28)).astype("float32") / 255

    # st.write("test image")
    # st.image(test_image, caption="test Image", width=150)
    # st.write(test_image_data)
    # # Display the raw canvas input
    # st.write("Raw Canvas Image (Grayscale):")
    # st.image(image, caption="Grayscale Image", width=150)

    # # Display the inverted image
    # st.write("Inverted Canvas Image:")
    # st.image(inverted, caption="Inverted Image", width=150)

    # # Display the resized and normalized image
    # st.write("Resized and Normalized Image:")
    # st.write(resized)  # Print raw pixel values
    # st.image(resized, caption="28x28 Input", width=150)
    # st.write(resized_data)
    if st.button("Predict"): 
        
    # # Predict the digit
        prediction = model.predict(resized_data)
        predicted_digit = prediction.argmax()
        confidence = prediction[0][predicted_digit]

        st.write(f"The model predicts: **{predicted_digit}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
