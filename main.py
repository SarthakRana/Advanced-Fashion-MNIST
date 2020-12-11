import time
import train
import constants
import preprocess
import warnings
import tensorflow as tf
from numpy.random import seed
from model import CustomCNNModel
warnings.filterwarnings('ignore')

# Seed Everything !!
seed(42)
tf.random.set_seed(42)


def main(start_time):
    # get preprocessed data
    DATA_PATH = constants.DIR
    print("[INFO]Data processing started...")
    train_processed_data, test_processed_data = preprocess.preprocess_data(DATA_PATH)
    print("[INFO]Data Processing complete...")
    
    # Build the CNN model
    print("[INFO]Building CNN model...")
    model = CustomCNNModel.buildModel(constants.IMG_DIMS[0], constants.IMG_DIMS[1])

    # Train model
    print("[INFO]Model training started...")
    trainedModel = train.trainCNN(model, train_processed_data, test_processed_data)
    print("[INFO]Model training complete...")
    
    # Save the trained model in CWD
    print("[INFO]Saving model...")
    trainedModel.save("CustomCNNModel")
    print("[INFO]Model saved successfuly...")
    
    # Calculate the execution time for entire code
    end_time = time.time()
    total_time = end_time - start_time
    print(f"TOTAL EXECUTION TIME OF PROGRAM : {total_time} secs")

if __name__ == "__main__":
    start_time = time.time()
    main(start_time)