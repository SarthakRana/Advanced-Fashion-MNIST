import constants
import tensorflow as tf
from numpy.random import seed
from tensorflow.keras import Input  
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten

# Seed Everything !!
seed(42)
tf.random.set_seed(42)


class CustomCNNModel:
    def colorNet(inputs):
        colornet = Conv2D(16, (3,3), padding="same")(inputs)
        colornet = Activation("relu") (colornet)
        colornet = BatchNormalization() (colornet)
        colornet = MaxPooling2D((3,3)) (colornet)
        colornet = Dropout(0.25) (colornet)
        colornet = Conv2D(32, (3,3), padding="same") (colornet)
        colornet = Activation("relu") (colornet)
        colornet = BatchNormalization() (colornet)
        colornet = MaxPooling2D((2,2)) (colornet)
        colornet = Dropout(0.25) (colornet)
        colornet = Conv2D(32, (3,3), padding="same") (colornet)
        colornet = Activation("relu") (colornet)
        colornet = BatchNormalization() (colornet)
        colornet = MaxPooling2D((2,2)) (colornet)
        colornet = Dropout(0.25) (colornet)
        colornet = Flatten() (colornet)
        colornet = Dense(128) (colornet)
        colornet = Activation("relu") (colornet)
        colornet = BatchNormalization() (colornet)
        colornet = Dropout(0.25) (colornet)
        colornet = Dense(64) (colornet)
        colornet = Activation("relu") (colornet)
        colornet = BatchNormalization() (colornet)
        colornet = Dropout(0.5) (colornet)
        colornet = Dense(constants.NUM_COLORS) (colornet)
        colornet = Activation("softmax", name="color_output") (colornet)
        return colornet
    
    def categoryNet(inputs):
        categorynet = Lambda(lambda x : tf.image.rgb_to_grayscale(x))(inputs)
        categorynet = Conv2D(32, (3,3), padding="same")(categorynet)
        categorynet = Activation("relu") (categorynet)
        categorynet = BatchNormalization() (categorynet)
        categorynet = MaxPooling2D((3,3)) (categorynet)
        categorynet = Dropout(0.25) (categorynet)
        categorynet = Conv2D(64, (3,3), padding="same") (categorynet)
        categorynet = Activation("relu") (categorynet)
        categorynet = BatchNormalization() (categorynet)
        categorynet = MaxPooling2D((2,2)) (categorynet)
        categorynet = Dropout(0.25) (categorynet)
        categorynet = Conv2D(128, (3,3), padding="same") (categorynet)
        categorynet = Activation("relu") (categorynet)
        categorynet = BatchNormalization() (categorynet)
        categorynet = MaxPooling2D((2,2)) (categorynet)
        categorynet = Dropout(0.25) (categorynet)
        categorynet = Flatten() (categorynet)
        categorynet = Dense(256) (categorynet)
        categorynet = Activation("relu") (categorynet)
        categorynet = BatchNormalization() (categorynet)
        categorynet = Dropout(0.25) (categorynet)
        categorynet = Dense(128) (categorynet)
        categorynet = Activation("relu") (categorynet)
        categorynet = BatchNormalization() (categorynet)
        categorynet = Dropout(0.5) (categorynet)
        categorynet = Dense(constants.NUM_CATEGORIES) (categorynet)
        categorynet = Activation("softmax", name="category_output") (categorynet)
        return categorynet
    
    def buildModel(height, width):
        inputShape = (height, width, 3) 
        inputs = Input(shape=inputShape)
        
        colorClassifier = CustomCNNModel.colorNet(inputs)
        categoryClassifier = CustomCNNModel.categoryNet(inputs)
        
        model = Model(
            inputs = inputs,
            outputs = [colorClassifier, categoryClassifier]
        )
        return model