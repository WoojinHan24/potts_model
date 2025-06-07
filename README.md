# Potts Model Classification and Tensor Renormalization Group

The repository aims to obtain the critical phenomena of the potts model in several methods.
The system currently have python module, its dependencies are controlled by poetry.
Please install poetry to follow up the package needed automatically.


## file system
[python library](/libs)
[source codes](/src)

## how to run the code?
in the python kernal, install poetry
```
pip install poetry
```
install the python environment on your computer.
if you want to create a virtual environment in conda, you may activate the environment first.
Now, install all the dependencies for this repository by run
```
poetry install
```
now run run.py in source code by
```
poetry run square_trg
```
or, for logging
```
poetry run square_trg &> logs/run.log
```
