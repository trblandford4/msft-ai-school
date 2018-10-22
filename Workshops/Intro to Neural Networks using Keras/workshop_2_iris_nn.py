# iris_nn.py
# Anaconda 5.2.0 Python 3.6.5
# TensorFlow 1.10.0 Keras 2.2.2
# Ray Blandford 
# Sept 20, 2018 - AI School Lect. 2

import numpy as np
import keras as K
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
    # 0. get started
    print("\nIris dataset using Keras/TensorFlow \n")
    np.random.seed(4)
    tf.set_random_seed(1)

    # 1. load data
    print("Loading all Iris data into memory \n")
    data_file = ".\\Data\\iris_data.txt"
    # Note: np function to convert txt file to a Float32
    train_x = np.loadtxt(data_file, usecols=[0,1,2,3], delimiter=",", skiprows=0, dtype=np.float32)
    train_y = np.loadtxt(data_file, usecols=[4,5,6], delimiter=",", skiprows=0, dtype=np.float32)

    # 2. define model
    model = K.models.Sequential()
    # Note: 5 hidden layers, uses tanh for range [-1,1] to avoid bias in gradients
    model.add(K.layers.Dense(units=5, input_dim=4, activation='tanh'))
    # Note: classification problem uses softmax for activation function
    model.add(K.layers.Dense(units=3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # 3. train model
    print("Starting training \n")
    h = model.fit(train_x, train_y, batch_size=1, epochs=10, verbose=1)  # 1 = very chatty
    print("\nTraining finished \n")

    # 4. evaluate model
    eval = model.evaluate(train_x, train_y, verbose=0)
    print("Evaluation: loss = %0.6f  accuracy = %0.2f%% \n" % (eval[0], eval[1]*100) )

    # 5. save model
    # TODO

    # 6. use model
    np.set_printoptions(precision=4)
    unknown = np.array([[6.1, 3.1, 5.1, 1.1]], dtype=np.float32)
    predicted = model.predict(unknown)
    print("Using model to predict species for features: ")
    print(unknown)
    print("\nPredicted species is: ")
    print(predicted)

if __name__=="__main__":
  main()


