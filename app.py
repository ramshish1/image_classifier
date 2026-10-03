"""
Streamlit app: Cats vs. Dogs classifier.

Run with:
    streamlit run app.py

Expects a trained model saved at ./cats_dogs_model.keras (produced by the
cats_vs_dogs_cnn.ipynb notebook). Place this app.py in the same folder as
that model file, or change MODEL_PATH below.
"""
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "cats_dogs_model.keras"
IMG_SIZE = (150, 150)
CLASS_NAMES = ["cat", "dog"]  # alphabetical, matches keras.utils.image_dataset_from_directory

st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐾", layout="centered")


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image):
    arr = preprocess(image)
    score = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS_NAMES[1] if score > 0.5 else CLASS_NAMES[0]
    confidence = score if score > 0.5 else 1 - score
    return label, confidence, score


st.title("🐾 Cats vs. Dogs Classifier")
st.write(
    "Upload a photo of a cat or a dog, and the CNN trained in "
    "`cats_vs_dogs_cnn.ipynb` will predict which one it is."
)

try:
    model = load_model()
except Exception as e:
    st.error(
        f"Couldn't load the model from `{MODEL_PATH}`. "
        f"Make sure that file is in the same folder as this app.\n\nError: {e}"
    )
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Classifying..."):
            label, confidence, raw_score = predict(model, image)

        st.subheader("Prediction")
        st.markdown(f"## {'🐱' if label == 'cat' else '🐶'} {label.capitalize()}")
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(confidence)

        with st.expander("Raw model output"):
            st.write(f"Sigmoid score (P(dog)): {raw_score:.4f}")
            st.write(f"Threshold: 0.5  →  score > 0.5 means 'dog', otherwise 'cat'")

else:
    st.info("Upload a .jpg or .png image to get a prediction.")

st.divider()
st.caption(
    "Model: a 4-block CNN (Conv2D + BatchNorm + MaxPool, 32→64→128→128 filters) "
    "trained from scratch on the TensorFlow Cats vs Dogs sample dataset "
    "(2,000 training images). Not a production-grade classifier — accuracy "
    "depends on how well your uploaded photo resembles the training distribution."
)
