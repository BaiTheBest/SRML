# Environment Setup


## OCTAVE

migration to octave

install octave with homebrew (macos): 
```bash
brew install octave
```

or apt (linux):  
```bash
apt install octave
apt install liboctave-dev
```


to run octave:  
```bash
octave-cli -qf
```

install additional packages
```octave
pkg install -forge io
pkg install -forge statistics
pkg install -forge linear-algebra
```


install jupyter kernel in a python virtualenv
```bash
pip install virtualenv
pip install virtualenvwrapper
mkvirtualenv wmtl
pip install octave_kernel
python -m octave_kernel install --user
```

## install code
```bash
git clone https://github.com/johnnytorres/wmtl
git submodule init 
git submodule update
git pull --recurse-submodules
```


# AUC roc implementation
https://towardsdatascience.com/roc-curve-and-auc-from-scratch-in-numpy-visualized-2612bb9459ab

# initial configuration of the project 

## MALSAR
only the first time install MALSAR library for baseline methods
```
git submodule add https://github.com/jiayuzhou/MALSAR
```

## OPTUNITY

only at the project creation, installed modified version of optunity https://optunity.readthedocs.io/en/latest/user/installation.html#install-octave to support multitask datasets
**only works with python2**
```
git submodule add https://github.com/johnnytorres/optunity.git
```

## jsonlab
install jsonlab library to save json results 
```
git submodule add https://github.com/fangq/jsonlab.git
```

## Octave config
add paths for the installed libraries, change the path if it's in a different folder

```matlab
addpath(genpath(make_absolute_filename('MALSAR/MALSAR/functions'))); 
addpath(genpath(make_absolute_filename('MALSAR/MALSAR/utils'))); 
addpath(genpath(make_absolute_filename('optunity-multitask/wrappers/octave/optunity'))); 
addpath(make_absolute_filename('jsonlab'));
savepath()
```


## issues with file created in Windows OS
when trying to run from bash ./main.m
it raises an error : "'usr/local/bin/octave-cli: invalid option -- '..."

verify line ending
cat -e main.m

to run from command line with octave change line ending with:
perl -pi -e 's/\r\n/\n/' main.m   

to run with matlab change line ending with:
perl -pi -e 's/\n/\r\n/' main.m   

## segmentation fault issue 
dataset_path="/root/data/multitasktest/school/school.mat"
results_path="/root/data/multitasktest/school/results/mtreg_caso/test"
load (dataset_path)
test_size=0.3
%pkg load statistics  <-- the culprit ???
addpath('utils/'); 
[X_train, Y_train, X_test, Y_test] = datasplit(X, Y, test_size);
solver="grid_small";
pars=struct('rho1', 100, 'rho2', 100, 'k', 1);
Least_CASO(X, Y, pars.rho1, pars.rho2, pars.k, pars);

# DATASET 

TODO 
```
./scripts/data.sh 20news
```


# EXPERIMENTS

## regression task experiments 

supported models: mtreg_l21,mtreg_lasso,mtreg_caso,mtreg_rmtl,mtreg_wmtl

supported datasets: computerBuyers,04cars,school,trafficSP,parkinson,isolet,sarcos,solarFlare


```
DATASET=syntheticWMTLR1 
MODEL=mtreg_l21 
ITERATIONS=1
GRID_SIZE=grid_small
MIN_TRAIN_SIZE=0.5
MAX_TRAIN_SIZE=0.5

./scripts/experiments.sh ${DATASET} ${MODEL} ${ITERATIONS} ${GRID_SIZE} ${MIN_TRAIN_SIZE} ${MAX_TRAIN_SIZE}
```

## classification task experiments 

supported models: mtclf_lasso,mtclf_l21,mtclf_caso,mtclf_srmtl,mtclf_wmtl

supported datasets: unrestnlp,crisisnlp,webKB,20news


```
DATASET=unrestnlp 
MODEL=mtclf_lasso 
ITERATIONS=10
GRID_SIZE=grid_small
MIN_TRAIN_SIZE=0.2
MAX_TRAIN_SIZE=0.9

./scripts/experiments.sh ${DATASET} ${MODEL} ${ITERATIONS} ${GRID_SIZE} ${MIN_TRAIN_SIZE} ${MAX_TRAIN_SIZE}
```


### 

syntheticWMTLR4t20m100d25p0s0
syntheticWMTLR7t20m500d25p0s0
syntheticWMTLR8t100m100d1000p0s0


regression 
DATASETS=syntheticWMTLR4t20m100d25p0s0,school,04cars,computerBuyers,facebook,trafficSP,isolet
MODELS=mtreg_lrst,mtreg_lasso,mtreg_l21,mtreg_caso,mtreg_rmtl,mtreg_wmtl,mtreg_swmtl3,mtreg_swmtl31,mtreg_swmtl32

classification
DATASETS=syntheticWMTLC8t5m100d2p1s0,unrestnlp,crisisnlp,civilColombia,civilBrazil,civilMexico,civilParaguay,civilVenezuela
MODELS=mtclf_lasso,mtclf_l21,mtclf_caso,mtclf_srmtl,mtclf_swmtl32,mtclf_swmtl33




'''
MODELS=mtreg_swmtl33
DATASET=syntheticWMTLR8t20m100d25p1s0sp0
MODE=local
STORE=local
SOLVER=grid_small
ITERS=5
MIN_TRAIN_SIZE=0.6
MAX_TRAIN_SIZE=0.6
./scripts/run.sh $DATASET $MODELS $MODE $STORE $SOLVER $ITERS $MIN_TRAIN_SIZE $MAX_TRAIN_SIZE 
'''


'''
MODELS=mtclf_caso
DATASET=unrestParaguay
TEST_DATASET=unrestParaguay2014
MODE=local
STORE=local
SOLVER=grid_small
ITERS=1
MIN_TRAIN_SIZE=0.8
MAX_TRAIN_SIZE=0.8
./scripts/run.sh $DATASET $MODELS $MODE $STORE $SOLVER $ITERS $MIN_TRAIN_SIZE $MAX_TRAIN_SIZE $TEST_DATASET
'''