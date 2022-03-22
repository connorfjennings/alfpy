# alfpy
* under test 
* fitting with emcee and dynesty works
* scripts in `https://github.com/cconroy20/alf/tree/master/src' 
  are (almost) directly translated into python
* requires ALF_HOME and as alf, all models in alf/infiles/ are required
* use `python3 alf.py filename tag` to run
* `setup` is parallelized and will pickle a large model array 
* edit `tofit_parameters.py` to specify parameters to fit
* packages required: 
    - numba, numpy, pickle, emcee, dynesty, schwimmbad
