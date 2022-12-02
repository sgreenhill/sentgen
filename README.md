# Semantic Neural Latent Traversal

## About

This repository contains source code and data accompanying the paper "Semantic Control of Generative Musical Attributes" presented at ISMIR 2022:

> Stewart Greenhill, Majid Abdolshah, Vuong Le, Sunil Gupta, Svetha Venkatesh. "Semantic Control of Generative Musical Attributes", in Proceedings of the 23rd International Society for Music Information Retrieval Conference (ISMIR), 2022.

Semantic Neural Latent Traversal (aka "SeNT")  is an algorithm for nagivating the latent spaces of generative neural networks such as VAEs and GANs. Previous approaches to achieving controllability have focused on regularisation and disentanglement, so that each semantic attribute is linearly related to a dimension of the latent space. The SeNT method uses a secondary neural network to learn the relationship between semantic attributes and latent dimensions.

We demonstrate the SeNT method using the [dMelodies data-set](https://github.com/ashispati/dmelodies_dataset) which contains a database of algorithmically-generated two-bar melodies and their features. We also use a set of generative VAE models from the [dMelodies controllability](https://github.com/ashispati/dmelodies_controllability) paper. These code-bases are included as sub-modules of this Git repository. For technical reasons (see below) we are using forks of the original two repositories.

### 1) Preparation

* Install `anaconda` or `miniconda` as described [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). For reference, this software was developed using miniconda (version 4.10.3) with python 3.7.2.

* Re-running these experiments from scratch will require a lot of computing time on large machines. As a short-cut we have provided some pre-built data which can be used as described below. These large files (around 1.5Gb in the "prebuilt" directory)  are stored on Git-LFS. To retreive these files (and the original dmelodies data set) ensure that you install the Git LFS extension as described [here](https://git-lfs.github.com/). Otherwise, the repository will be retreived with pointers to these large files.

* Clone this repository using the `--recurse-submodules` flag, which will automatically include the required sub-modules: *dmelodies_dataset* and *dmelodies_controllability*.

* Follow the instructions in *dmelodies_controllability/README.md* for creating a dMelodies conda environment.

* Activate the dmelodies environment, and install the following additional packages:

```
conda activate dmelodies
pip install fire
```

There are three source directories that need to be in the PYTHON_PATH environment: src, dmelodies_controllability and dmelodies_controllability/dmelodies_dataset. To set this, do the following in the root directory:
```
source scripts/env.sh
```

The `scripts` directory contains shell scripts used to automate parts of the build process. Most of these tasks involve neural network training via pyTorch, and will be slow without GPU acceleration or multiple CPU cores. The script files below may be submitted to a [SLURM cluster](https://slurm.schedmd.com/) using the [sbatch](https://slurm.schedmd.com/sbatch.html) command. For example:
```
sbatch scripts/build-vae-models.sh
```

Please first review the included `#SBATCH` options and adjust for your local system. To run efficiently on a desktop machine you will need to reduce the level of parallelism.

### 2) Train the dMelodies VAE Models

We use the VAE models publshed in [dmelodies_controllability](https://github.com/ashispati/dmelodies_controllability). These should be trained using the original scripts. Our forked version modifies the process slightly, to allow more control of partitioning of the data between training, validation and testing. The original hard-codes a 71/20/10 split, but this does not leave enough data for training our secondary SeNT neural network model. We introduced the --split option to allow an alternative split.

To build the VAE models from scratch:

```
scripts/build-vae-models.sh
```

Rebuilding the 36 VAE models is a lengthy process. Alternatively, to install the pre-built VAE models:
```
tar xzf prebuilt/vae-models.tar.gz
```

### 3) Import latent codes  and features for SeNT Training

To train the SeNT models we need latent positions and their corresonding features, arranged in pairs that are exemplars of attribute changes. This process is done in two stages:

* Extract a set of latent codes and their corresponding features. This is done using the `dmgenerate` utility, which enumerates the testing samples for a specific model.

* Find sets of training pairs. This is done by processing the normalised features to identify pairs that represent good attribute changes.

To build this data from scratch:

```
scripts/make-training-data.sh
```

This is a lengthy process. Alternatively, to install the pre-built data:
```
tar xzf prebuilt/gen.tar.gz
```

### 4) Training and testing the SeNT Models

A separate traversal function is built for each feature of each model. For 36 models and 9 features this is 324 traversal functions. For each function we do the following steps:

1. Analyse mutual information between the feature and the latent dimensions.
2. Train a surrogate "Z to C" network for approximating the gradient of the feature with respect to Z
3. Train the neural traversal function. This uses the decoder of the original VAE model during evaluation of the perceptual loss term. This introduces a significant bottleneck in the training process, with the bulk of the time being spent in the underlying music21 functions that are used for dMelodies feature extraction. Feature extraction dominates the run-time so there is little benefit gained by GPU acceleration of the network training, but significant gain from parallel implementation on multiple CPU cores.
4. For a set of test attribute changes, predict the latent code and evaluate the corresponding feature value, again using the VAE decoder.

To run train and test the neural traversal functions using the default settings:
```
scripts/train-sent.sh
```

This should create a directory `data/run1e` which contains 36 sub-directories, one with the results each evaluated model.

This script calls `src/Evaluate.py` which manages the above tasks efficiently using parallel processes. We use 128 cores for this computation, but this can be adjusted using the `--jobs` option as described below. Other parameters can also be varied through command-line options.

* `--jobs` :  number of parallel jobs to evaluate simultaneously (default: 1)
* `--threads` : number of threads to allocate per job (default: 1)
* `--outDir` : specify output directory
* `--auto` : specify output directory. Certain characters in the name will activate particular options. Some of these are documented in `NeuralTraversal2.py`.
    * `e` : Use 80% of the available samples for training and 20% for testing ("extended"). Otherwise, the default is to use 32K for training, and 10K for testing.
    * `h`: Use one-hot encoding for feature values (experimental)
    * `s`: Rather than using the VAE decoder to generate the feature for the perceptual loss, use the "simulated" value predicted by the "Z to C" surrogate network. This approximation is *much* faster than using the decoder
    * `i`: Scale the internal network nodes to twice the number of dimensions, increasing the number of network parameters (experimental)
    * `p`: Add two additional dense stages in the SeNT network, increasing the number of network parameters (experimental)
    * `c`: Include the feature value C in addition to the normal `subtract` node in the SeNT network (experimental)

### 5) Performance Analysis and Post-processing

Analysis of the training results is done by `src/Perform.py` though some outputs are generated using other tools (gnuplot, gs, image-magick). The version of matplotlib used is incompatible with the numpy/pyTorch/matplotlib combination of the dmelodies environment, so this process must be run in a separate environment. To build the environment:
```
conda env create -f scripts/post/environment.yml
conda activate post
```

Add the directory `scripts/post` to the executable PATH, then in the `data` directory do:
```
post.sh run1e
```

Results in the form of pdf and csv files are generated in `data/run1e`.

[The results directory](results) includes some sample results, and an explanation of additional results that are not included in the original paper.

## Contents

This repository includes:

* `dmelodies_controllability` :  submodule containing VAE generative models and training scripts
    * `dmelodies_controllability/dmelodies_dataset` : sub-submodule containing dMelodies data set, and pyTorch data loader
* `src` : source code for SeNT model
    * `NeuralTraversal2.py` : implementation  and training of SeNT neural traversal function
    * `Perform.py` : performance analysis for testing
    * `Find_MI.py` : evaluate mutual information for loss functions in SeNT training
    * `util.py` and `build.py` : utility functions for creating and reading repository of training data
    * `dmgenerate.py` : generation of training data from dMelodies VAE models
    * `dmutil.py` : utility functionsfor handling dMelodies VAE models
    * `Evaluate.py` : main driver for training and testing SeNT models
    
* `prebuilt` : pre-built data sets
    * `vae-models.tar.gz` - 36 pre-trained VAE models used for evaluation of SeNT
    * `gen.tar.gz` - training data for SeNT models, including latent codes with their corresponding features, and lists of attribute changes
    
* `scripts` : assorted scripts used for training and evaluating the models
    * `post` : performance analysis and post-processing scripts

## Implementation Details

The dMelodies data contains 1354752 algorithmically-generated two-bar  melodies and their features. This data is stored in the file:
> dmelodies_controllability/dmelodies_dataset/data/dMelodies_dataset.npz

 which contains a `score_array` with 16 note values for two bars of quarter notes, and `latent_array` with the features for each sample. This file is roughly 130Mb in size, and if `git lfs` has not been used will be a pointer file. If the data file is not present, follow the instructions in dmelodies_controllability/dmelodies_dataset/README.md to generate the dataset, or install git lfs to retreive the original file.
 
 The dMelodies data is partitioned into training/validation/testing subsets. By default the allocation is 70/20/10 percent of the samples in each subset. Since the "testing" subset must be sufficient for additional training and testing of the SeNT models we change the default allocation to 70/5/25, which yields 338688 samples for training and testing the SeNT models. 
 
 Four VAE model types are used here: beta-VAE, ar-VAE, interp-VAE, s3-VAE. For each of these types, we train three hyper-parameter settings with three random seeds, a total of 36 models. The trained VAE models are stored in
 > dmelodies_controllability/src/saved_models-70-05-25
 
 which includes a serialised pyTorch neural network for each model. Note that this naming scheme is slightly different to the original, to allow for multiple VAE model sets with different partitioning of the data.
 
To train the SeNT models, we extract the latent codes and features from the VAE "testing" data. This data is stored in a ".npz" file of roughly 64Mb, which includes a 338688 x 32 "latent_codes" array, and a 338688 x 9 "attributes" array. The resulting data set is about 2.4Gb in size.

We then search the training data for pairs of samples that represent good exemples of attribute changes. This process can be quite time consuming since it is O(N<sup>2</sup>) for a N data points. We use two methods to optimise the search:

* The attribute changes depends only on the attributes, and not the latent codes. Attribute sequencing is determined by the choice of random seed. Thus for the 36 models, we only need to compute one sequence of pairs for each attribute for each of the three random seeds, a total of 27 sequences.
* Computation of pairs turns out to be slow when implemented in Python (see build.py:gen). Instead the procedure is implemented in C (see findpairs.c).
* Training pairs are free of repeats, so may be arbitrarily partitioned between training and testing subsets.

The training data for the next stage is stored in:

> data/gen
 
 Each SeNT model is stored in a separate "repository" directory. A repository is initialised in util.py:init2 by specifying a source file, pairs file, and a partitioning of the pairs between training and testing sets. This generates a config.json file which includes the following attributes:
 
 * `source` is the path to the file containing the original latent codes and attributes
 * `pairs` is the path the the file containing the indices of the samples for the attribute change pairs
 * `srange` is the range information used to normalise latent codes to [-1,1]
 * `frange` is the range information used to normalise the features to [0,1]
 * `fnames` are the names of the features
 * `ttPairs` are the indices in the `pairs` sequences corresponding to training / testing subsets for each feature
 
 The main implementation for the SeNT training is in NeuralTraversal2.py. Probably the most complex aspect of this training is the computation of the perceptual loss term. Given a latent code we must determine the resulting feature so we can evaluate its deviation from `c2`, the intended feature value. We use the decoder of the original VAE model to get the value of the feature (via `decodeFeatures`) but to successfuly backpropagate this loss term we also need its gradient. Because of the complexity of this term its gradient is unavailable through the pyTorch auto-differentiation framework. Therefore we use separate "Z to C" neural network as a surrogate to approximate the gradient. This is trained on the original data, but will be smoother than the true gradient since it doesn't see invalid states that are sometimes generated by the VAE decoder.
 
Many data files are generated during the training and testing of SeNT models. These include:

* `corr_index_F.csv` and `uncorr_index_F.csv` are the correlated and uncorrelated latent dimensions for feature F, computed from mutual information
* `sent_F.model` is the pyTorch neural network model for the SeNT traversal function
* `sent_ZC_F.model` is a separate neural network used as a surrogate for the gradient of C
* `sp_F.csv` are the predicted latent codes for the testing data
* `z2_F.csv` are reference latent codes for the attribute target. These may legitimately be different to the predicted values
* `cp_F.csv` are the attributes for the predicted target latent code, determined using the VAE decoder


