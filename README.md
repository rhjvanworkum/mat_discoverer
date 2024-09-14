# mat_discoverer

## 1. Installation
- Make sure that you have either Anaconda or Miniconda installed
- `git clone https://github.com/rhjvanworkum/mat_discoverer`
- `conda env create -f env.yml`
- One will have to make slight adjustments to the orb_models package, namely:
    - go into ~/miniconda3/envs/mat2/lib/python3.11/site-packages/orb_models/forcefield/featurization_utilities.py and comment out line 9 (which imports pynanoflann) and on line 222 change 'pynanoflann' to 'scipy'

## 2. How to run the algorithm
- Make sure to create an API Key for the Materials project and copy this one in the env.sh file
- `source env.sh`
- Run the algorithm like this: `python bin/run_evolutionary_optimization_algorithm.py --species "['Zn', 'Cu']" --num_species "[1, 1]" --population_size 15`

