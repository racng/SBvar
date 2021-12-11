# sbvar: Varying Parameter Analysis for Systems Biology
[![Build Status](https://app.travis-ci.com/racng/SBvar.svg?branch=main)](https://app.travis-ci.com/racng/SBvar)
[![license](https://img.shields.io/github/license/racng/SBvar.svg?style=flat-square)](https://github.com/racng/SBvar.svg/main/LICENSE)

Sbvar is a Python package for performing robustness analysis to characterize how the dynamics of biological networks respond to changes in parameters. User can design one-way experiment to vary one parameter or two-way experiments to vary two parameters over any range of values. For every condition, sbvar finds the steady states or dynamic time series of species concentrations, species derivatives, and reaction rates. These responses' relationships with the varying parameter(s) can be further visualized by 2D or 3D plots.

Sbvar integrates existing packages that specializes in simulations (tellurium and roadrunner), data structure and transformations (numpy, pandas, anndata), and visualization (matplotlib). Models are loaded using SBML or antimony input via tellurium. 

## General workflow of analysis:
1. Load reaction model using tellurium.
2. Create experiment by specifying the model, parameter(s) to change, and the associated range of values. Examples of parameters:
    - Floating species initial concentration 
    - Boundary species concentration
    - Reaction constants
    - Hill coefficients
    - Other variables defined in the model

3. For each condition, we can simulate/calculate the following for species concentration (and derivatives) and reaction rates:
    - Time series
    - Steady State
4. Visualization of timeseries and steady states.
5. Flexible downstream analysis on time series based on custom functions.  

# Usage Example
See [notebook](https://github.com/racng/SBvar/blob/main/notebook/bistable_system.ipynb) for detailed examples for one-way and two-way experiment on a bistable system.
## Step 1:
Loading an antimony model as a roadrunner object using tellurium. 
```Python
import sbvar as sb
import tellurium as te

ant = '''
    J0: $Xo -> S1; 1 + Xo*(32+(S1/0.75)^3.2)/(1 +(S1/4.3)^3.2);
    J1: S1 -> $X1; k1*S1;

    Xo = 0.09; X1 = 0.0;
    S1 = 0.5; k1 = 3.2;
'''
rr = te.loada(ant)
```

## Step 2:
Create 1D experiment where we vary the initial concentration of S1 from 0 to 10 using 40 evenly spaced samples. We specify simulations to start from t=0 and end at t=4, using 100 evenly spaced timepoints. 

```Python
exp = sb.experiment.OneWayExperiment(rr, param='S1', bounds=(0,12), num=40, 
    start=0, end=4, points=100, conserved_moiety=True)
```
## Step 3:
Simulate timeseries and calculate steady states.
```Python
exp.simulate()
exp.calc_steady_state()
```
## Step 4:
Visualize time series of S1 concentration across all conditions as a filled contour plot or 3D surface. 
```Python
exp.plot_timecourse_mesh('S1', levels=20)
exp.plot_timecourse_mesh('S1', kind='surface', projection='3d')

```

# Installation

Install from PyPI:
```
pip install sbvar
```
Install from github:
```
git clone https://github.com/racng/SBvar.git
cd SBvar
pip install .
```

# Authors
- Rachel Ng ([@racng](https://github.com/racng))