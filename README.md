## Lung disease classification with Pytorch


The repository benchmarks various pre-trained models to reveal the best performing model for the task of lung disease classification. A simple CNN is used as a baseline. The used dataset: ChestX-ray14.

### Results

TO BE WRITTEN

### What the final web interface looks like

TO BE WRITTEN

### How to run the project

1. run in the terminal `pip install -r requirements.txt` to download the library requirements.

2. run in terminal `python main.py` to download the models with obtianed weights, see the analysis for each model and the framework for each of the models. To re-train the models yourself run the following command in terminal: `python main.py --re_train True`. WARNING: re-training will take significant time (12+ hours) and space (60 GB) due to the dataset size.

### System requirements

For using the models without re-training: 1 GB CUDA and 1 minute of execution / no CUDA and 5 minutes of execution.

For using the models with re-training: 10 GB CUDA 