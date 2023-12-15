import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
model = load_model('model.h5')

def main():
    st.title("Drowsiness Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        result = model.predict(preprocess_image(image))

        st.write("Prediction:", "Drowsy" if result > 0.5 else "Not Drowsy")

def preprocess_image(image):
    # Resize the image to 145x145 pixels
    resized_image = cv2.resize(image, (145, 145))

    normalized_image = resized_image / 255.0

    preprocessed_image = np.expand_dims(normalized_image, axis=0)

    return preprocessed_image


if __name__ == '__main__':
    main()
