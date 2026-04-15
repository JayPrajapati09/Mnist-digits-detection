# Mnist-digits-detection

Worked on Popular MNIST dataset to recognize digits.

### Models : 
Created Three different models for image classification.

* Model 1 : Simple architecture 
  -> Layer 1 : conv - ReLU - conv - ReLU - MaxPool
  -> Layer 2 : conv - ReLU - conv - ReLU - MaxPool
  -> Layer 3 : conv - ReLU - MaxPool
  -> Layer 4 : flattten


* Model 2 : Deep  architecture 
  -> Layer 1 : conv - ReLU - conv - ReLU - conv - ReLU - conv - ReLU - MaxPool
  -> Layer 2 : conv - ReLU - conv - ReLU - conv - ReLU - MaxPool
  -> Layer 3 : conv - ReLU - conv - ReLU - MaxPool
  -> Layer 4 : flattten


* Model 3 : Mobile Net
 Used Transfer learning


### Performance after 10 epochs


| Model| Train Accuracy | Test Accuracy | 
| -------- | -------- | -------- |
| Simple Model(vo)    | 99.26     | 99.01 |
| Deep Model (v1)   | 99.70     | 99.41 |
| Transfer Learning(v2) |98.47| 98.31 | 

# Streamlit APP:
[Mnist APP](https://mnist-digits-classification.streamlit.app/)
