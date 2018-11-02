## Introduction

List of Python programs to be executed on PC in order to train convulted neural network.

1. [a_singleround_train_cnn.py](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/blob/master/05_keras_cnn_train/00_pc_side/a_singleround_train_cnn.py)
    - Take the training data and train model weight to get the desirable output. Although it is pre-setted with 20 epoch rounds, it will stop halfway when the trained model weight stop getting better at guessing test data to prevent overfitting and only the best result will be saved in 01_model_weight folder.
2. [b_multiround_train_cnn.py](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/blob/master/05_keras_cnn_train/00_pc_side/b_multiround_train_cnn.py)
    - Similar to [a_singleround_train_cnn.py](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/blob/master/05_keras_cnn_train/00_pc_side/a_singleround_train_cnn.py) but it will repeat the training with clean TensorFlow graph.