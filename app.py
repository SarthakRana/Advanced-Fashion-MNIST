import cv2
import pickle
import numpy as np
from PIL import Image
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
st.set_option('deprecation.showfileUploaderEncoding', False)

IMG_DIMS = (96,96,3)    # h,w,c
NUM_COLORS=3    # Black, Blue, Red
NUM_CATEGORIES=4    # Dress, Jeans, Shirt, Shoes

# Header
st.markdown("<h1 align='center' style='background-color:tomato'>👗 Advanced Fashion MNIST 👖</h1>", unsafe_allow_html=True)
st.markdown("<h2 align='center'>Welcome to Advanced Fashion MNIST web app!</h2>", unsafe_allow_html=True)
st.markdown("<h3 align='center'><i>A machine learning approach to detect the category and color of clothes</i></h3>", unsafe_allow_html=True)

# About Project and How it Works
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h3>⚙️ How it works</h3>", unsafe_allow_html=True)
st.write("* Advanced Fashion MNIST is about detecting the category and color of clothes.")
st.write("* Upload an image of the cloth/ person wearing that cloth.")
st.write("* Click on `Detect` and wait for a few seconds for program to load model, process data, etc.")
st.write("* You will see the predictions (along with class accuracy).")
st.write("* You will also see a plot showing the model predictions for each class.")

st.warning("**NOTE** : *Currently, the app is capable of detecting the following categories and colors:*\n\n" +
           "**Colors** *: Black, Blue, Red*\n\n" + 
           "**Categories** *: Dress, Jeans, Shirt, Shoes*"
          )

st.markdown("<br>", unsafe_allow_html=True)
X_user=[]
color_labels = ['Black', 'Blue', 'Red']
category_labels = ['Dress', 'Jeans', 'Shirt', 'Shoes']
st.markdown("<h4>Choose an image for classification...</h4>", unsafe_allow_html=True)
uploaded_file_buffer = st.file_uploader(label="", type="jpg")
if uploaded_file_buffer is not None:
    img = Image.open(uploaded_file_buffer) # Pillow reads image in RGB format by default
    st.image(img, width=300)
    img = np.array(img)
    img = cv2.resize(img, (IMG_DIMS[1], IMG_DIMS[0]))
    img = img / 255.0
    X_user.append(img)
    X_user = np.array(X_user)
    if st.button("Detect"):
        # Predict color and category labels
        st.write("[INFO] Loading model...")
        model = load_model("CustomCNNModel")
        st.write("[INFO] Classifying...")
        (color_probs, category_probs) = model.predict(X_user)
        
        # get max prob indices and max prob values
        color_idx = color_probs.argmax()
        cat_idx = category_probs.argmax()
        color_prob = color_probs[0][color_idx]
        cat_prob = category_probs[0][cat_idx]
        
        # inverse trasform colors and category to get label values
        lb_color = pickle.loads(open("labelBinarizer_color.pkl", "rb").read())
        lb_category = pickle.loads(open("labelBinarizer_category.pkl", "rb").read())
        color_arr = np.zeros((1, NUM_COLORS))
        cat_arr = np.zeros((1, NUM_CATEGORIES))
        color_arr[0][color_idx] = 1
        cat_arr[0][cat_idx] = 1
        col = lb_color.inverse_transform(color_arr)
        cat = lb_category.inverse_transform(cat_arr)
        st.write("Done!")
        
        st.success("**Color** : " + str(col[0]) + " [" + str(round(color_prob*100, 3)) + "]\n\n" +
                   "**Category** : " + str(cat[0]) + " [" + str(round(cat_prob*100, 3)) + "]"
                   )
        
        # Plotting probabilities
        fig, ax = plt.subplots(1,2, figsize=(12, 5))
        
        ax[0].set_title("Color prediction probabilities")
        sns.barplot(color_labels, color_probs[0], ax=ax[0])
        ax[0].set_xlabel("Colors")
        ax[0].set_ylabel("Probabilities")
        
        ax[1].set_title("Category prediction probabilities")
        sns.barplot(category_labels, category_probs[0], ax=ax[1])
        ax[1].set_xlabel("Colors")
        ax[1].set_ylabel("Probabilities")
        
        st.pyplot(fig)