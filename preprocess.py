import os
import cv2
import pickle
import warnings
import constants
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from numpy.random import seed
warnings.filterwarnings('ignore')
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

# Seed Everything !!
seed(42)
tf.random.set_seed(42)

def preprocess_data(DATA_PATH):
    """
    The idea here is to read the files, rename them and save them with new names. 
    Default file name are just numbers(i.e. 00001, 00023, etc.). We will add label 
    to them (eg:- red_dress_234, blue_jeans_23, etc.) so as to make it easier for 
    us to fetch and input labels while training.
    """
    # STEP 1 : Read folder names from directory
    print("[INFO]Reading all image files...")
    files = []
    for files_ in os.listdir(DATA_PATH):
        files.append(files_)
             
        
    # STEP 2 : Maintain a list of all images path
    src_img_list = []
    for folder in os.listdir(DATA_PATH):
      temp_dir = DATA_PATH + folder + '\\'
      for imgs in os.listdir(temp_dir):
        src_img_list.append(temp_dir + imgs)
    shuffle(src_img_list, random_state=42)
    
    
    # STEP 3 : Split data into train and test set(train:80% | test:20%)
    print("[INFO]Splitting data into train and test set...")
    train_len = int(len(src_img_list) * 0.8)
    train_data = src_img_list[0 : train_len]
    test_data = src_img_list[train_len : ]
    
    # STEP 4 : Load image, resize it, convert color from BGR to RGB and separate color and category labels
    def get_data_for_model(data):
      X = []
      y_color = []
      y_category = []
      for path in tqdm(data):
        img = cv2.imread(path)
        img = cv2.resize(img, (constants.IMG_DIMS[1], constants.IMG_DIMS[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 reads image in BGR format
        color_category = path.split('\\')[-1].split('_')
        (color, category) = color_category[0:2]
        
        # Normalize the image pixel values by bringing values in range 0 to 1
        img = img / 255.0
    
        X.append(img)
        y_color.append(color)
        y_category.append(category)
      
      return np.array(X) , np.array(y_color), np.array(y_category)
    
    # Get train data and label for model training 
    print("[INFO]Processing train data...")
    X_train , y_color_train, y_category_train = get_data_for_model(train_data)
    
    # Get test data and label for model testing
    print("[INFO]Processing test data...")
    X_test, y_color_test, y_category_test = get_data_for_model(test_data)
    
    # Label Binarize the color and category output for model training and testing purpose
    lb_color = LabelBinarizer()
    lb_category = LabelBinarizer()
    
    y_color_train = lb_color.fit_transform(y_color_train)
    y_category_train = lb_category.fit_transform(y_category_train)
    
    y_color_test = lb_color.transform(y_color_test)
    y_category_test = lb_category.transform(y_category_test)
    
    train_processed_data = [X_train, y_color_train, y_category_train]
    test_processed_data = [X_test, y_color_test, y_category_test]
    
    # Save category label binarizer for classification purpose
    f = open(constants.SAVE_DIR + 'labelBinarizer_category.pkl', "wb")
    f.write(pickle.dumps(lb_category))
    f.close()
    
    # Save color label binarizer classification purpose
    f = open(constants.SAVE_DIR + 'labelBinarizer_color.pkl', "wb")
    f.write(pickle.dumps(lb_color))
    f.close()
    
    return train_processed_data, test_processed_data