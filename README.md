# alfpy
* under test 
* scripts in `https://github.com/cconroy20/alf/tree/master/src' 
  are (almost) directly translated into python
* requires ALF_HOME and everything in alf/infiles/
* use `python3 alf.py filename tag` to run
* `setup` is parallelized and will pickle a large model array 
* use global variable `use_keys` to specify parameters to fit
* speed up by numba
* packages required: 
    - numba, numpy, pickle, emcee, dynesty, schwimmbad


