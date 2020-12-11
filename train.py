import warnings
import constants
import tensorflow as tf
from numpy.random import seed
warnings.filterwarnings('ignore')
from tensorflow.keras.callbacks import EarlyStopping

# Seed Everything !!
seed(42)
tf.random.set_seed(42)

# Define early stopping callback
def getEarlyStopper():
  earlyStopper = EarlyStopping(monitor='val_loss', 
                               patience=10, 
                               restore_best_weights=True,
                               verbose=1, 
                               mode='auto'
                               )
  return earlyStopper


def trainCNN(model, train_processed_data, test_processed_data):
    
    X_train = train_processed_data[0]
    y_color_train = train_processed_data[1]
    y_category_train = train_processed_data[2]
    X_test = test_processed_data[0]
    y_color_test = test_processed_data[1]
    y_category_test = test_processed_data[2]
    lr = constants.LR
    nb_epochs = constants.NB_EPOCHS
    batch_size = constants.BATCH_SIZE
       
    optimizer = tf.keras.optimizers.Adam(lr=lr, decay=lr/nb_epochs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    y_train = {
        'category_output' : y_category_train,
        'color_output' :  y_color_train
    }
    y_test = {
        'category_output' : y_category_test,
        'color_output' :  y_color_test
    }
    model.fit(X_train, y_train,
              epochs=nb_epochs, 
              batch_size=batch_size,
              callbacks=[getEarlyStopper()],
              validation_data=(X_test, y_test),
              verbose=1
              )   
    
    return model