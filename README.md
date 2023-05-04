# Lung disease classification with Pytorch

### Dataset & preprocessing

The repository benchmarks various pre-trained models to reveal the best performing model for the task of lung disease classification. The used dataset: ChestX-ray14, which consists of 112k 1024x1024 pixel x-ray images of 14 distinct lung diseases as well as images of healthy individuals. 

What the preprocessed images look like:

<img src="./assets/data.png" width="700" height="350">

### System requirements

For using the models without re-training: 1 GB CUDA and 1 minute of execution / no CUDA and 5 minutes of execution.

For using the models with re-training: 10 GB CUDA 
