# Training Keras with convulted neural network to drive

## Introduction

<p align = "center">
  <img src = "https://raw.githubusercontent.com/hafiz-kamilin/autonomous_pixhawk_rover/master/05_keras_cnn_train/keras_cnn.png" width = 800" height = "500"/>
</p>

Take training data recorded from [03_collect_train_data](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/tree/master/03_collect_train_data) or generated from [04_train_data_doubler](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/tree/master/04_train_data_doubler) and use it to train convulted neural network. Neural network then will classify the test images as moving forward, turning left and turning right. Optimizers can be freely choosen and only the best trained model weight with lowest val_loss (validation loss) value will be saved in 01_model_weight folder.

If you want to test the program without preparing the training data, you can get the sample from [here](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/releases/tag/v1.2).

## Guide

First, download required Python programs [00_pc_side](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/tree/master/05_keras_cnn_train/00_pc_sidee) to PC. After that copy 00_training_data folder recorded from [03_collect_train_data](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/tree/master/03_collect_train_data) or generated from [04_train_data_doubler](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/tree/master/04_train_data_doubler) into the same directory as [a_singleround_train_cnn.py](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/blob/master/05_keras_cnn_train/00_pc_side/a_singleround_train_cnn.py) and [b_multiround_train_cnn.py](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/blob/master/05_keras_cnn_train/00_pc_side/b_multiround_train_cnn.py) Python program.

After that install keras, tensorflow, scikit-learn, panda, pandas, pillow, pydot and matplotlib Python packages on PC. In addition to visualize neural network model, [Graphviz](https://www.graphviz.org/) also need to be installed and set the binary in system PATH. This might not be a complete list of packages or software needed to run these programs, so pay attention to the errors generated in order to find hints of where things goes wrong.

## Training model weight

Simply execute [a_singleround_train_cnn.py](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/blob/master/05_keras_cnn_train/00_pc_side/a_singleround_train_cnn.py) Python program and follow the instructions. The best trained model weight will be saved in 01_model_weight folder. 

If the val_acc (validation accuracy) is too low or the best result can not be replicated use [b_multiround_train_cnn.py](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/blob/master/05_keras_cnn_train/00_pc_side/b_multiround_train_cnn.py) and set how many time it should repeat the training with clean TensorFlow graph. It take longer time to complete but the result is near replicable. Same as [a_singleround_train_cnn.py](https://github.com/hafiz-kamilin/autonomous_pixhawk_rover/blob/master/05_keras_cnn_train/00_pc_side/a_singleround_train_cnn.py) only the best trained model weight will be saved.

## Troubleshooting

Sometime when training the model weight it will suddenly stuck during epoch round, this might be a bug caused by TensorFlow backend used by Keras. To solve this simply click on the Command Prompt or PowerShell and press [Enter] key to unstuck it.
