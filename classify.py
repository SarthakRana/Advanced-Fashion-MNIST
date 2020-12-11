import cv2
import pickle
import argparse
import constants
import numpy as np
from tensorflow.keras.models import load_model

X_user = [] 
actual_values = []

# Construct an argument parser to take in necessary inputs from user
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="provide path to saved model")
parser.add_argument("-i", "--image", required=True, help="provide path of image which you want to classify")
parser.add_argument("-lbcol", "--colorbin", required=True, help="path to color label binarizer")
parser.add_argument("-lbcat", "--catbin", required=True, help="path to category label binarizer")
args = vars(parser.parse_args())

# Read the user image, resize it, normalize the pixel values and covert to numpy array
img = cv2.imread(args["image"])
img = cv2.resize(img, (constants.IMG_DIMS[1], constants.IMG_DIMS[0]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 reads image in BGR format
img = img / 255.0
X_user.append(img)
X_user = np.array(X_user)

# Predict color and category labels
model = load_model(args["model"])
(color, category) = model.predict(X_user)

# get max prob indices and max prob values
color_idx = color.argmax()
cat_idx = category.argmax()
color_prob = color[0][color_idx]
cat_prob = category[0][cat_idx]

# inverse trasform colors and category to get label values
lb_color = pickle.loads(open(args["colorbin"], "rb").read())
lb_category = pickle.loads(open(args["catbin"], "rb").read())
color_arr = np.zeros((1, constants.NUM_COLORS))
cat_arr = np.zeros((1, constants.NUM_CATEGORIES))
color_arr[0][color_idx] = 1
cat_arr[0][cat_idx] = 1
col = lb_color.inverse_transform(color_arr)
cat = lb_category.inverse_transform(cat_arr)

print("\n<====== CNN MODEL PREDICTION ======>\n")
print("Predicted Color : {} ({:.2f})".format(col[0], color_prob*100))
print("Predicted Category : {} ({:.2f})".format(cat[0], cat_prob*100))