from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "trained_plant_model2.keras"
HOME_IMAGE_PATH = BASE_DIR / "home_page.jpeg"
IMG_SIZE = (128, 128)  # matches the model's input shape (128, 128, 3)

# 38 classes, same order as the model's softmax output
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]


def pretty_name(raw: str) -> str:
    """'Tomato___Late_blight' -> 'Tomato - Late blight'"""
    plant, _, condition = raw.partition("___")
    return f"{plant.replace('_', ' ').strip()} - {condition.replace('_', ' ').strip()}"


# Load the model once and reuse it across reruns/clicks
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# TensorFlow model prediction
def model_prediction(uploaded_file):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=IMG_SIZE)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)  # raw 0-255 pixels, as the model expects
    input_arr = np.expand_dims(input_arr, axis=0)  # single image -> batch of 1
    prediction = model.predict(input_arr, verbose=0)[0]
    result_index = int(np.argmax(prediction))
    confidence = float(prediction[result_index])
    return result_index, confidence


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    if HOME_IMAGE_PATH.exists():
        st.image(str(HOME_IMAGE_PATH), width="stretch")
    else:
        st.warning(f"Home image not found at: {HOME_IMAGE_PATH}")
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. Train (70295 images)
    2. Valid (17572 image)
    3. Test (33 images)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, width="stretch")

    if st.button("Predict"):
        if test_image is None:
            st.warning("Please upload an image first.")
        elif not MODEL_PATH.exists():
            st.error(f"Model file not found at: {MODEL_PATH}")
        else:
            with st.spinner("Please Wait.."):
                result_index, confidence = model_prediction(test_image)
            st.write("Our Prediction")
            st.success(
                f"Model is predicting it's **{pretty_name(CLASS_NAMES[result_index])}** "
                f"({confidence:.1%} confidence)"
            )