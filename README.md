# alfpy
* under test 
* scripts in `https://github.com/cconroy20/alf/tree/master/src' 
  are (almost) directly translated into python
* fitting with emcee and dynesty works
* works with MPIpool on clusters
* requires ALF_HOME and as alf, all models in alf/infiles/ are needed
* 1. `python3 alf_build_model.py filename tag`
* 2. `python3 alf.py filename tag`
* `setup` is parallelized and will pickle a large model array 
* edit `tofit_parameters.py` to specify parameters to fit
* packages required: 
    - numba, numpy, pickle, emcee, dynesty, schwimmbad
