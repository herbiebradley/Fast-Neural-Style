Fast-Neural-Style
==============================

A Pytorch implementation of [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).

Requirements:

- Pytorch 1.0
- Python 3.6

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md
    ├── data
    │   └── raw            <- Raw data before any processing
    │
    ├── saved_models       <- Checkpointed models and tensorboard summaries
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project
        ├── __init__.py    <- Makes src a Python module
        │
        ├── train.py       <- Run this to train
        │
        ├── test.py        <- Run this to test
        │
        ├── pipeline       <- Code for downloading or loading data  
        │
        ├── options        <- Files for command line options
        │
        ├── models         <- Code for defining the network structure and loss functions
        │
        └── utils          <- Utility files, including scripts for visualisation

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
