# Packages Required

import tensorflow as tf
import csv
import numpy as np
import pickle
import base64
import boto3
from logger_config import logger

# Functions required

####    Function Call #####

####  windowed dataset ####
def windowed_dataset(client_data, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(client_data)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

####  SimpleMLP ##

class SimpleMLP:
    @staticmethod
    def build():
        tf.random.set_seed(51)
        np.random.seed(51)
        model = tf.keras.models.Sequential([ tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                           input_shape=[None]), tf.keras.layers.SimpleRNN(400, return_sequences=True),
                                           tf.keras.layers.SimpleRNN(400), tf.keras.layers.Dense(1), ])
        return model

def download_file_from_s3(bucket_name, object_name, local_file_path):
    s3_client = boto3.client("s3",
                      region_name="eu-north-1")
    try:
        logger.info(f"Downloading file from bucket '{bucket_name}' with object '{object_name}' and storing to path '{local_file_path}'")
        s3_client.download_file(bucket_name, object_name, local_file_path)
        return True
    except Exception as e:
        logger.info(f"Error downloading file '{object_name}' from bucket '{bucket_name}': {e}")
        return False
def local_training(global_weights):
    Mbits_transmitted = []
    with open('./Traffic_Train_Data.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            Mbits_transmitted.append(float(row[1]))

    # converting the lists into arrays
    series_trans = np.array(Mbits_transmitted)
    window_size = 20
    batch_size = 20
    total_epochs = 5
    dataset = windowed_dataset(np.array(series_trans), window_size, batch_size, len(series_trans))
    smlp_local = SimpleMLP()
    local_model = smlp_local.build()
    logger.info("compiling the model with a mean squared error loss and a stochastic gradient descent optimizer")
    local_model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=3e-4, momentum=0.9),
                        metrics=["mae"])
    logger.info("set local model weight to the weight of the global model")
    local_model.set_weights(global_weights)
    logger.info(f"fit local model with client's data with Epochs - {total_epochs}, Batch size - {batch_size}, Window size - {window_size}")
    local_model.fit(dataset, epochs=total_epochs, verbose=0)
    serialized_weights = pickle.dumps(local_model.get_weights())
    base64_encoded_weights = base64.b64encode(serialized_weights).decode('utf-8')
    return 'clientSTHLM', len(series_trans), base64_encoded_weights
