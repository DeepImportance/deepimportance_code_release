## Setting up DeepImportance

One can reach the source code of our implementation via terminal command

    git clone https://github.com/DeepImportance/deepimportance_code_release.git

Running this code requires `python2.7` to be installed. Check [this](https://www.python.org/downloads/)
website for installation.

Also, there are multiple libraries needed. We strongly recommend you to create
a virtual environment and install packages in this environment.
For creating a virtual environment check
[this](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
webpage.

Once the virtual environemnt is set, you can install required libraries by using
`pip package manager` as shown below. If `pip` is not installed on your
computer first install it as described
[here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
or [here](https://pip.pypa.io/en/stable/installing/).

    pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.0-py2-none-any.whl (OSX) 
    pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-py2-none-any.whl (Linux) 
    pip install Keras==2.2.2
    pip install numpy
    pip install scikit-learn
    pip install scikit-image
    pip install matplotlib
    pip install pillow
    pip install cleverhans

Once all the packages are installed you are ready to run DeepImportance code.
