import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained lung cancer prediction model
# Replace 'your_model.h5' with the path to your trained model file
model = tf.keras.models.load_model("C:/Users/rosha/Downloads/my_model.h5")

# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to the desired input shape (e.g., 128x128 pixels)
    desired_size = (128, 128)
    resized_image = image.resize(desired_size)

    # Convert the resized image to a NumPy array
    image_array = np.array(resized_image)

    # Ensure the image has 3 color channels (RGB)
    if image_array.shape[-1] != 3:
        # Convert grayscale to RGB by duplicating the single channel
        image_array = np.stack((image_array,) * 3, axis=-1)

    # Normalize the image pixel values (if needed)
    image_array = image_array / 255.0

    # Expand the dimensions to match the model's input shape
    input_data = np.expand_dims(image_array, axis=0)
    return input_data

# Streamlit app
st.title("Lung Cancer Prediction")

# Upload an image for prediction
uploaded_image = st.file_uploader("Upload an image for cancer prediction", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image and make a prediction
    input_image = Image.open(uploaded_image)
    processed_image = preprocess_image(input_image)
    prediction = model.predict(processed_image)

    # Determine the prediction result
    if prediction[0][0] > 0.5:
        st.write("Prediction: Lung Cancer Detected")
    else:
        st.write("Prediction: No Lung Cancer Detected")

    # Show the prediction probability
    st.write(f"Probability: {prediction[0][0]:.2%}")
