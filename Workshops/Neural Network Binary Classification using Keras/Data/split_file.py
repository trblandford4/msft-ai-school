# split_file.py
# does not read source into memory
# useful when no processing/normalization needed

import numpy as np

def file_len(fname):
 f = open(fname)
 for (i, line) in enumerate(f): pass
 f.close()
 return i+1

def main():
  source_file = ".\\cleveland_norm.txt"
  train_file = ".\\cleveland_train.txt"
  validate_file = ".\\cleveland_validate.txt"
  test_file = ".\\cleveland_test.txt"

  N = file_len(source_file)
  num_train = int(0.60 * N)
  num_validate = int(0.20 * N)
  num_test = N - (num_train + num_validate) # ~20%

  np.random.seed(1)
  indices = np.arange(N)  # array [0, 1, . . N-1]
  np.random.shuffle(indices)

  train_dict = {}
  test_dict = {}
  validate_dict = {}
  for i in range(0,num_train):
    k = indices[i]; v = i  # i is not used
    train_dict[k] = v

  for i in range(num_train,(num_train+num_validate)):
    k = indices[i]; v = i
    validate_dict[k] = v  

  for i in range((num_train+num_validate),N):
    k = indices[i]; v = i
    test_dict[k] = v 

  f_source = open(source_file, "r")
  f_train = open(train_file, "w")
  f_validate = open(validate_file, "w")
  f_test = open(test_file, "w")

  line_num = 0
  for line in f_source:
    if line_num in train_dict: # checks for key
      f_train.write(line)
    elif line_num in validate_dict:
      f_validate.write(line)
    else:
      f_test.write(line)
    line_num += 1

  f_source.close()
  f_train.close()
  f_validate.close()
  f_test.close() 

if __name__ == "__main__":
  main()
