# alfpy
__alfpy__ is a Python translation of the Absorption Line Fitter [__alf__](https://github.com/cconroy20/alf/tree/master/src). The original Fortran code from [__alf__](https://github.com/cconroy20/alf/tree/master/src) has been almost directly translated into Python, with some improvements.

## Overview
I started this project in 2021 with the goal of getting a comprehensive understanding of how the __alf__ code operates. I’m happy to say that I have achieved that goal, and now the Python version, __alfpy__, not only functions well but also offers faster performance and flexibility (e.g., changing number of parameters instead of simply shrinking their priors) in many cases.

## Key Features and Differences from the Original Fortran Version
- Samplers: __alfpy__ supports both emcee and dynesty samplers.
- Performance: The code has been partially accelerated using numba, and parallelization is achieved with Python’s multiprocessing module.
- Dependencies: __alfpy__ requires all the models from the original __alf__ project, located under `alf/infiles/`.

## Installation and Requirements
To run alfpy, you’ll need the following Python packages:
- numpy
- numba
- pickle
- emcee
- dynesty
- multiprocessing

## Usage Instructions
1.	Edit `tofit_parameters.py` to specify the parameters you want to fit.
2.	Run the following command to build the model:
 `python3 alf_build_model.py <filename> <tag>`
3.	To start fitting the model, run:
`python3 alf.py <filename> <tag>`

## Citation
If alfpy is helpful in your work, I kindly ask that you cite this GitHub repository.
