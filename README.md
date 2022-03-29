# alfpy
* This is a python version of absorption line fitter (alf). 
  The code in `https://github.com/cconroy20/alf/tree/master/src' 
  is almost directly translated. 
* Available samplers are emcee and dynesty. The code is partially accelerated by numba, parallelized by multiprocessing.
* alfpy requires all models in alf (under alf/infiles/)
* To use:
	* 1. edit `tofit_parameters.py` to specify parameters to fit
	* 2. `python3 alf_build_model.py filename tag`
	* 3. `python3 alf.py filename tag` 
* packages required: 
    - numpy, numba, pickle, emcee, dynesty, multiprocessing
