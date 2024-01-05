# Classification_app.py
import numpy as np
import pandas as pd
import copy
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from functions import *

# Retrieve data from the path
train_X, train_Y, test_X, test_Y, classes = load_and_process_dataset()

# Train the model
logistic_regression_model = model(train_X, train_Y, test_X, test_Y, num_iterations=2000, learning_rate=0.005, print_cost=True)

st.title("Image classification.")
st.write("Upload an image.")
up_file = st.file_uploader("")

if up_file is not None:
    pict = Image.open(up_file)
    image = np.array(pict)
    plt.imshow(image)
    image = image.reshape((1, num_px * num_px * 3)).T
    image = image / 255.
    my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)
    st.write("Algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
    st.write("Test Accuracy is " + str(logistic_regression_model['test_accuracy']) + "%")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
