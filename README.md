# Importance-Driven Deep Learning System Testing -ICSE 2020

### About
This paper [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3628024.svg)](https://doi.org/10.5281/zenodo.3628024)
presents DeepImportance, a systematic testing methodology accompanied by an Importance-Driven (IDC)
test adequacy criterion for DL systems. Applying IDC enables to
establish a layer-wise functional understanding of the importance
of DL system components and use this information to guide the
generation of semantically-diverse test sets. Our empirical evaluation on several DL systems, across multiple DL datasets and with
state-of-the-art adversarial generation techniques demonstrates the
usefulness and effectiveness of DeepImportance and its ability to
guide the engineering of more robust DL systems.

### Repository
This repository includes details about the artifact corresponding to implementation of DeepImportance.
Our implementation is publicly available in
[DeepImportance repository](https://github.com/DeepImportance/deepimportance_code_release).
This artifact allows reproducing the experimental results presented in the paper. Below we
describe how to reproduce results. Before going further, first, check
installation page (i.e. INSTALL.md).

### Installation
This version of deep importance runs with Python 2.7. Make sure you are working in an 
environment with python 2.7 installed. Checkout 'Setting up DeepImportance' in install.md file for more 
detail.

To install the dependencies, run the command below:

`pip install -r requirements.txt`

For more information on the dependencies, check out "install.md".

### Sample Usage

Deep Importance repository contains models Lenet1, Lenet4, Lenet5, and Dave under the
'neural_networks' folder.

```
# To see the list of commands

python run.py -H

# Run importance driven coverage

python run.py -A idc

```