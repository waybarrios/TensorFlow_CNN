#CNN TensorFLow
CIFAR-10 is a common benchmark in machine learning for image recognition.
Link:
http://www.cs.toronto.edu/~kriz/cifar.html

--------------------
Overview
--------------------

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

--------------------
Installation Ubuntu/Linux
--------------------

```
$ pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```
--------------------
Model Architecture
--------------------


The model in this CIFAR-10 tutorial is a multi-layer architecture consisting of alternating convolutions and nonlinearities. These layers are followed by fully connected layers leading into a softmax classifier. The model follows the architecture described by Alex Krizhevsky, with a few differences in the top few layers.

This model achieves a peak performance of about 86% accuracy within a few hours of training time on a GPU. Please see below and the code for details. It consists of 1,068,298 learnable parameters and requires about 19.5M multiply-add operations to compute inference on a single image.

--------------------
Code Organization
--------------------

File                        | Purpose
----------------            | -------------
cifar10_input.py            | Reads the native CIFAR-10 binary file format.
cifar10.py                  | Builds the CIFAR-10 model.
cifar10_train.py            | Trains a CIFAR-10 model on a CPU or GPU.
cifar10_multi_gpu_train.py  | Trains a CIFAR-10 model on multiple GPUs.
cifar10_eval.py             | Evaluates the predictive performance of CIFAR10


--------------------
CIFAR-10 Model
--------------------
The CIFAR-10 network is largely contained in ```cifar10.py```. The complete training graph contains roughly 765 operations. We find that we can make the code most reusable by constructing the graph with the following modules:


    1. Model inputs: inputs() and distorted_inputs() add operations that read and preprocess CIFAR images for evaluation and training, respectively.
    
    2. Model prediction: inference() adds operations that perform inference, i.e. classification, on supplied images.
    
    3. Model training: loss() and train() add operations that compute the loss, gradients, variable updates and visualization summaries.

--------------------
Model Inputs
--------------------
The input part of the model is built by the functions ```inputs()``` and ```distorted_inputs()``` which read images from the CIFAR-10 binary data files. These files contain fixed byte length records, so we use ```tf.FixedLengthRecordReader```.

The images are processed as follows:


    * They are cropped to 24 x 24 pixels, centrally for evaluation or randomly for training.
    * They are approximately whitened to make the model insensitive to dynamic range.

For training, we additionally apply a series of random distortions to artificially increase the data set size:


    * Randomly flip the image from left to right.
    * Randomly distort the image brightness.
    * Randomly distort the image contrast.

--------------------
Model Predicition
--------------------
  The prediction part of the model is constructed by the ```inference()``` function which adds operations to compute the logits of the predictions. That part of the model is organized as follows:

  Layer Name       | Description
  ---------------- | -------------
  conv1            | convolution and rectified linear activation.
  pool1            | max pooling.
  conv2            | convolution and rectified linear activation.
  norm2            | local response normalization.
  pool2            | max pooling.
  local3           | fully connected layer with rectified linear activation.
  local4           | fully connected layer with rectified linear activation.
  softmax_linear   | linear transformation to produce logits.

--------------------
Model Training
--------------------

The usual method for training a network to perform N-way classification is multinomial logistic regression, aka. softmax regression. Softmax regression applies a softmax nonlinearity to the output of the network and calculates the cross-entropy between the normalized predictions and a 1-hot encoding of the label. For regularization, we also apply the usual weight decay losses to all learned variables. The objective function for the model is the sum of the cross entropy loss and all these weight decay terms, as returned by the ```loss()``` function.

--------------------
Launching and Training the Model
--------------------
We have built the model, let's now launch it and run the training operation with the script cifar10_train.py.
```
python cifar10_train.py
```
You should see the output:
```
Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
2015-11-10 10:45:45.927302: step 0, loss = 4.68 (2.0 examples/sec; 64.221 sec/batch)
2015-11-10 10:45:49.133065: step 10, loss = 4.66 (533.8 examples/sec; 0.240 sec/batch)
2015-11-10 10:45:51.397710: step 20, loss = 4.64 (597.4 examples/sec; 0.214 sec/batch)
2015-11-10 10:45:54.446850: step 30, loss = 4.62 (391.0 examples/sec; 0.327 sec/batch)
2015-11-10 10:45:57.152676: step 40, loss = 4.61 (430.2 examples/sec; 0.298 sec/batch)
2015-11-11 10:46:00.437717: step 50, loss = 4.59 (406.4 examples/sec; 0.315 sec/batch)
```
The script reports the total loss every 10 steps as well the speed at which the last batch of data was processed. A few comments:

    * The first batch of data can be inordinately slow (e.g. several minutes) as the preprocessing threads fill up the shuffling queue with 20,000 processed CIFAR images.

    * The reported loss is the average loss of the most recent batch. Remember that this loss is the sum of the cross entropy and all weight decay terms.

    * Keep an eye on the processing speed of a batch. The numbers shown above were obtained on a Tesla K40c. If you are running on a CPU, expect slower performance.
    cifar10_train.py periodically saves all model parameters in checkpoint files but it does not evaluate the model. The checkpoint file will be used by cifar10_eval.py to measure the predictive performance (see Evaluating a Model below).

If you followed the previous steps, then you have now started training a CIFAR-10 model. Congratulations!

The terminal text returned from ```cifar10_train.py``` provides minimal insight into how the model is training.

--------------------
Evaluating a Model
--------------------
Let us now evaluate how well the trained model performs on a hold-out data set. the model is evaluated by the script  ```cifar10_eval.py ```. It constructs the model with the  ```inference() ``` function and uses all 10,000 images in the evaluation set of CIFAR-10. It calculates the precision at 1: how often the top prediction matches the true label of the image.

To monitor how the model improves during training, the evaluation script runs periodically on the latest checkpoint files created by the  ```cifar10_train.py. ```
```
python cifar10_eval.py
```

You should see the output:
```
2015-11-11 11:15:44.391206: precision @ 1 = 0.860
```

--------------------
Launching and Training the Model on Multiple GPU cards  
--------------------

If you have several GPU cards installed on your machine you can use them to train the model faster with the ```cifar10_multi_gpu_train.py``` script. It is a variation of the training script that parallelizes the model across multiple GPU cards.

```
python cifar10_multi_gpu_train.py --num_gpus=2
```

--------------------
Contributions
--------------------
http://tensorflow.org/tutorials
