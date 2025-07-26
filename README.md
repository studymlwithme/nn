# A from scratch neural network (fully-connected) framework using just numpy.. (Working AND a work in a progress)
My goal was to make the code self-documenting. But, in case it is not, here are some hints to get started. 

### The purpose of this library (for me) was to deeply understand how neural networks work. And that is indeed what it did for me. If you notice anything that seems wrong or needs correction feel free to open an issue or DM me, I know this code isn't perfect and I still have a lot to learn :) 



## Start: Main.py contains an example usage.. 
The basic set up of this framework is as seen by the main.py file.. 



  NN = Network(learning_rate=.01) ~ Initialize your network.
  
  NN.add_input_layer(X_train_std) ~ Add A0 (input layer w/ samples)
  
  NN.add_hidden_layer(X_train_std.shape[1], 13, "relu") --- (Hidden Layer #1)
  
  NN.add_hidden_layer(13, 12, "relu")  --- (Hidden Layer #2)
  
  NN.add_hidden_layer(12, 11, "relu")  --- (Hidden Layer #2)
  
  NN.add_hidden_layer(11, 10, "relu")  --- (Hidden Layer #2)
  
  NN.add_output_layer(10, 10, y_train, True, "softmax")   --- (Output Layer)
  
  NN.add_loss("negative_log_loss", y_train)  --- (Loss)
  


## Some important notes about the current state of this library.. 
#1. You must add a input layer prior to any hidden layers. 
#2. You can add as many hidden layers as you want, ensure that features_in of l match the features_out of l-1. 
#3. You must add an ouput layer (currently), as I have only created support for softmax + NLL as an output layer/loss combo. I will add support for binary cross entropy for sigmoid outputs.
#4. TanH is within the library, but it is not tested/validated, so it may break the network. 



