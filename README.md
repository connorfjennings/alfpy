# alfpy
__alfpy__ is a Python translation of the Absorption Line Fitter [__alf__](https://github.com/cconroy20/alf/tree/master/src). The original Fortran code from [__alf__](https://github.com/cconroy20/alf/tree/master/src) has been almost directly translated into Python, with some improvements.

## Overview
I started this project in 2021 with the goal of getting a comprehensive understanding of how the __alf__ code works. I’m happy to say that I have achieved that goal, and now the Python version, __alfpy__, not only mirrors the function but also offers faster performance and flexibility (e.g., changing number of parameters instead of simply shrinking their priors) in many cases.

I thank Charlie Conroy for his guidance on alf since the very beginning and sharing his epertise on its every detail.  I thank Josh Speagle for invaluable discussions on parameter convergence, and for introducing me to dynesty and optimizers like differential evolution.  I thank Aliza Beverage for helpful suggestions, implementing convergence tests and an automatic check of the acceptance fraction.  

## Key Features and Differences from the Original Fortran Version
- Samplers: __alfpy__ supports both emcee and dynesty samplers.
- Performance: The code has been partially accelerated using numba, and parallelized with multiprocessing.
- Dependencies: __alfpy__ requires all the models from the original __alf__ project, located under `alf/infiles/`.

## Installation and Requirements
To run alfpy, you’ll need the following Python packages (I list the version I use)
- numpy (1.26.4)
- numba (0.60.0)
- pickle
- emcee (3.1.6)
- dynesty (2.1.4)
- multiprocessing

## Usage Instructions
1. Define an environment variable `ALFPY_HOME` to `path/alfpy/`
2. Edit `tofit_parameters.py` to specify the parameters you want to fit and the default values for those not being fitted
3. With `<filename>.dat` placed in `alf/indata/`, run the following command to build the model:
 `python alf_build_model.py <filename> <tag>`
4. To start fitting the model, run:
`python alf.py <filename> <tag>`

## Citation
If __alfpy__ is helpful in your work, please kindly cite this GitHub repository, as well as all the relevant citations for the original [__alf__](https://github.com/cconroy20/alf/tree/master/src) which are mentioned in its documentation.
