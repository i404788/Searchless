# Searchless [pytorch_optimizer](https://github.com/kozistr/pytorch_optimizer) benchmark

This project tests the training of all pytorch optimizers with a sweep across common hyperparameters.
For each dataset-model-optimizer combination it will report the optimal hyperparameters with the margin of error (i.e. the gap between searched areas) for each param.

## Requirements
* diskcache (for intermediate results)
* pytorch
* pytorch_optimizer
* torchvision
* torchtext
