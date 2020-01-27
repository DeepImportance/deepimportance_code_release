# Importance-Driven Deep Learning System Testing -ICSE 2020

### About
This paper [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3628024.svg)](https://doi.org/10.5281/zenodo.3628024)
presents DeepImportance, a systematic testing methodology accompanied by an Importance-Driven (IDC)
test adequacy criterion for DL systems. Applying IDC enables to
establish a layer-wise functional understanding of the importance
of DL system components and use this information to guide the
generation of semantically-diverse test sets. Our empirical evalua-
tion on several DL systems, across multiple DL datasets and with
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

### Reproducing Results
Clone the repository via terminal command
    git clone https://github.com/DeepImportance/deepimportance_code_release.git

Some of the data (e.g. adversarial inputs etc.) is already provided in repository since it takes too much time to generate.

##### Reproducing results in Table 2 and Figure 4 in paper:
    cd deepimportance_code_release/reproduce/
    python TABLE2_FIGURE4.py

##### Reproducing results in Table 3 in paper:
    cd deepimportance_code_release/reproduce/
    python TABLE3_MNIST.py --approach idc --rel_neurons 6
    python TABLE3_MNIST.py --approach idc --rel_neurons 8
    python TABLE3_MNIST.py --approach idc --rel_neurons 10
    python TABLE3_MNIST.py --approach nc
    python TABLE3_MNIST.py --approach kmnca
    python TABLE3_MNIST.py --approach snac
    python TABLE3_MNIST.py --approach nbc
    python TABLE3_MNIST.py --approach lsc
    python TABLE3_MNIST.py --approach dsc

Same parameters for the following scripts:
    python TABLE3_CIFAR.py
    python TABLE3_DAVE.py

##### Reproducing results in Table  4 in paper:
    cd deepimportance_code_release/reproduce/
    python TABLE4.py --approach idc --rel_neurons 6
    python TABLE4.py --approach idc --rel_neurons 8
    python TABLE4.py --approach nc
    python TABLE4.py --approach kmnc
    python TABLE4.py --approach snac
    python TABLE4.py --approach nbc
    python TABLE4.py --approach lsc
    python TABLE4.py --approach dsc


##### Reproducing results in Table 5 in paper:
    cd deepimportance_code_release/reproduce/
    python TABLE5.py


