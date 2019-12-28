##Fast-Neural-Style
==============================

A Pytorch implementation of [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).

Created to learn PyTorch and mess around with best practices.

####Requirements:

You can run the Dockerfile by downloading the file on its own and using `docker build .`. Alternatively use the requirements.txt, but the PyTorch version can be 1.0+.

####Running

For the dataset I used the MSCOCO 2014 Training images dataset [80K/13GB] [(download)](http://mscoco.org/dataset/#download) - the same dataset used in Johnson et al. Required command line arguments for `train.py` are `cuda`, `data_dir`, and `save_model_dir`. Required command line arguments for `stylize.py` are `cuda`, `content_image`, `output_image`, and `model`. Use `train.py --help`  or `stylize.py --help` for more details.

Example training command:

```
python src/train.py --cuda 1 --datasets/train2014 --save_model_dir saved_models/checkpoints --style_image wave_crop.jpg
```

####Project Organization
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

