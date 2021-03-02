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


### Notes
* If you use Python 3.8, Cleverhans doe not yet support Tensforflow 2.x, so you should make a change at utils_tf.py
    def kl_with_logits(p_logits, q_logits, scope=None, loss_collection=tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES):

   See [https://github.com/cleverhans-lab/cleverhans/issues/1183](https://github.com/cleverhans-lab/cleverhans/issues/1183)
    


### Updates

* **02/03/21**: Updated with support for Python 3.8
