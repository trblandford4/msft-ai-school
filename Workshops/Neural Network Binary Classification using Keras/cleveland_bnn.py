# cleveland_bnn.py
# Anaconda3 5.2.0 (Python 3.6.5)
# TensorFlow 1.10.0  Keras 2.2.2  h5py 2.8.0

# ==================================================================================

import numpy as np
import keras as K
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class MyLogger(K.callbacks.Callback):
  def __init__(self, n):
    self.n = n 

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.n == 0:
      t_loss = logs.get('loss')
      t_accu = logs.get('acc')
      v_loss = logs.get('val_loss')
      v_accu = logs.get('val_acc')
      print("epoch = %4d  t_loss = %0.4f  t_acc = %0.2f%%  v_loss = %0.4f  \
v_acc = %0.2f%%" % (epoch, t_loss, t_accu*100, v_loss, v_accu*100))

# ==================================================================================

def main():
  # 0. get started
  print("\nCleveland binary classification dataset using Keras/TensorFlow ")
  np.random.seed(1)
  tf.set_random_seed(2)

  # 1. load data
  print("Loading Cleveland data into memory \n")
  train_file = ".\\Data\\cleveland_train.txt"
  valid_file = ".\\Data\\cleveland_validate.txt"
  test_file = ".\\Data\\cleveland_test.txt"

  train_x = np.loadtxt(train_file, usecols=range(0,18),
   delimiter="\t",  skiprows=0, dtype=np.float32)
  train_y = np.loadtxt(train_file, usecols=[18],
    delimiter="\t", skiprows=0, dtype=np.float32)

  valid_x = np.loadtxt(valid_file, usecols=range(0,18),
   delimiter="\t",  skiprows=0, dtype=np.float32)
  valid_y = np.loadtxt(valid_file, usecols=[18],
    delimiter="\t", skiprows=0, dtype=np.float32)

  test_x = np.loadtxt(test_file, usecols=range(0,18),
   delimiter="\t",  skiprows=0, dtype=np.float32)
  test_y = np.loadtxt(test_file, usecols=[18],
    delimiter="\t", skiprows=0, dtype=np.float32)

  # 2. define model
  init = K.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=1)
  simple_adadelta = K.optimizers.Adadelta()
  X = K.layers.Input(shape=(18,))
  net = K.layers.Dense(units=10, kernel_initializer=init,
    activation='relu')(X)
  net = K.layers.Dropout(0.25)(net)  # dropout for layer above
  net = K.layers.Dense(units=10, kernel_initializer=init,
    activation='relu')(net) 
  net = K.layers.Dropout(0.25)(net)  # dropout for layer above
  net = K.layers.Dense(units=1, kernel_initializer=init,
    activation='sigmoid')(net)
  model = K.models.Model(X, net) 

  model.compile(loss='binary_crossentropy', optimizer=simple_adadelta,
    metrics=['acc'])

  # 3. train model
  bat_size = 8
  max_epochs = 2000
  my_logger = MyLogger(int(max_epochs/5))

  print("Starting training ")
  h = model.fit(train_x, train_y, batch_size=bat_size, verbose=0,
    epochs=max_epochs, validation_data=(valid_x,valid_y),
    callbacks=[my_logger])
  print("Training finished \n")

  # 4. evaluate model
  eval = model.evaluate(test_x, test_y, verbose=0)
  print("Evaluation on test data: loss = %0.4f  accuracy = %0.2f%% \n" \
    % (eval[0], eval[1]*100) )

  # 5. save model
  print("Saving model to disk \n")
  mp = ".\\Models\\cleveland_model.h5"
  model.save(mp)

  # 6. use model
  unknown = np.array([[0.75, 1, 0, 1, 0, 0.49, 0.27, 1, -1, -1, 0.62, -1, 0.40,
    0, 1, 0.23, 1, 0]], dtype=np.float32)
  predicted = model.predict(unknown)
  print("Using model to predict heart disease for features: ")
  print(unknown)
  print("\nPredicted (0=no disease, 1=disease) is: ")
  print(predicted)

# ==================================================================================

if __name__=="__main__":
  main()
