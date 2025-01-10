# Learnable Fractal Flames
This repository contains the code associated with the project [Learnable Fractal Flames](https://arxiv.org/abs/2406.09328). This work presents a differentiable rendering approach that allows latent fractal flame parameters to be learned from image supervision. 

![teaser](https://github.com/user-attachments/assets/c71ef339-2fec-4465-b4cc-d594faac5e79)

## Setup/Install
To run the examples, it is recommended to use a virtual python environment https://docs.python.org/3/library/venv.html#module-venv. 
Run the following command to create a venv.

    $ python -m venv venv

Within a virtual environment, the package can be installed with pip.
If you intend to edit/develop the package, it is easiest to use a dev installation so that you do not need to re-install after every change.
https://setuptools.pypa.io/en/latest/userguide/development_mode.html 

Run the following command from the root project directory to install dfractal in development mode:

    $ pip install -e .

After installation, you should be able to import the dfractal library and run the examples within the virtual enviroment where the package was installed.
