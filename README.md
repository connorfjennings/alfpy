# alfpy
* scripts in `https://github.com/cconroy20/alf/tree/master/src' 
  are (almost) directly transted into python
* requires ALF_HOME and everything in alf/infiles/
* both emcee and dynesty are supported, just change `run` in alf.py
* need to pickle a large model array (about 3hrs dep on imf_type and data) 
  (based on setup.f90) 
* use global variable use_keys to specify parameters to fit
