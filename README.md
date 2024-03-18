
# DPNeurifyFV

Neural network verification based on input splitting and forward propagation of symbolic intervals with fresh variables.

## Installation

Note: `DPNeurifyFV` is currently only supported for x86 architectures.

Open the Julia REPL in this directory and type
- `]` to open the Julia package manager
- `add https://github.com/sisl/NeuralVerification.jl#0d9be34` (adds `NeuralVerification.jl` to the environment, but pinned to a specific commit)
    - at least use the pinned version, until [this issue](https://github.com/sisl/NeuralVerification.jl/issues/201) with the installation of `NeuralVerification.jl` is resolved
- `add https://github.com/sisl/NeuralPriorityOptimizer.jl` and follow the installation instructions on [their repo](https://github.com/sisl/NeuralPriorityOptimizer.jl) 
    - install [Gurobi](https://www.gurobi.com/) (academic licenses are free)
    - type `install Gurobi`
    - (to use all features of `NeuralPriorityOptimizer`, you also need to install Mosek, however, this is not necessary for `DPNeurifyFV` or the experimental evaluation.)
- `add LazySets`
- `add https://github.com/phK3/DPNeurifyFV.jl`

## Examples

In order to maximize the first output of the `ACAS_1_1` network over the input region defined by the first property of the ACAS-Xu benchmark suite run

```julia
using NeuralVerification, DPNeurifyFV

input_set, output_set = DPNeurifyFV.get_acas_sets(1)

acas = read_nnet("./networks/ACASXU_experimental_v2a_1_1.nnet")

params = DPNeurifyFV.PriorityOptimizerParameters(max_steps=5000, print_frequency=100, stop_frequency=1, verbosity=2)
optimize_linear_deep_poly(acas, input_set, [1.,0,0,0,0], params, solver=DPNFV(method=:DeepPolyRelax, max_vars=15), concrete_sample=:BoundsMaximizer, split=DPNeurifyFV.split_important_interval)
```

Further examples can be found in `./example_notebook.ipynb`.

## Reproducing Experiments

To reproduce the experiments in the paper, open the Julia REPL in this directory and type
- `]` to open the package manager
- `activate . ` (to activate the DPNeurifyFV environment)
- press backspace to leave the package manager
- type `include("experiments/run_experiments.jl")` to start verification of the ACAS Xu benchmark set using DPNeurifyFV
    - timeouts etc. can be changed in the script
- type `include("experiments/zope_experiments.jl")` to start verification with ZoPE
- to reproduce the results for `NNENUM`, 
    - download `NNENUM` from GitHub: https://github.com/stanleybak/nnenum
    - use the Python script under `experiments/run_nnenum.py` and place it in `nnenum/examples/acasxu/`
    - run the script 


## Creating Sysimage

In order to avoid long loading times for the package, we can generate a precompiled sysimage of `DPNeurifyFV` by executing 
```julia
using Pkg
Pkg.activate(".")
include("vnncomp_scripts/create_sysimage.jl")
```
The sysimage will be created as `./sys_dpneurifyfv.so` (in the root directory of `DPNeurifyFV`).
The compilation can take long itself (it took ~10mins on my laptop), but subsequent calls are fast, when the sysimage is used:
```
julia --sysimage ./sys_dpneurifyfv.so 
```
starts a Julia session with the sysimage loaded.


## Todo LSTM Solver

- [x] include bounds of output layer in LP, s.t. we get monotonically decreasing loss
- [x] Use solution of LP for propagating concrete input through the NN
- [x] Split LSTM functions at origin first
- [ ] find better splitting for LSTM functions after origin split
- [x] implement Remez-like algorithm instead of sampling-based LP
- [x] use memoization to avoid recomputing all of the LPs

### LSTM Solver Development

- We get too many splits that probably don't really help in reducing the overapproximation: We need less splits (do we really need 4 subsets every time?) and we need to improve the splitting point

#### Better Plan for Splitting

1) Use function approximation that doesn't require linear regression (e.g. Chebyshev approximation or Galerkin method, ...)
2) Find best splitting point along one variable by just proposing some splitting point and calculating linear approximation on both halves along with their error. Take the splitting point which minimizes the error on both sides (this is just finding the minimum of a one dimensional function)
3) Ultimately the LSTM activation function is 5-dimensional, so do this for each of the five variables

#### Overapproximations for LSTM Functions

- Overapproximations calculated via LPs are vastly superior to overapproximations given by linear least squares regression, but a single forward pass is then also more than 10x slower!
- added *Remez-like* algorithm that doesn't just sample some points and then fits the minimax linear approximation via LP, but starts from the corner points, fits the LP (with only those 4 points), then only adds points where the maximum error is greater than the LP estimate. In this way the number of constraints is way smaller than with the sampling LP and the process is deterministic.
    - runtime improved by about 2x
    - bounds also significantly improved (also roughly 2x)
- added memoization to cache calls to the LP solver (if we split in layer $n$, we don't need to recompute the relaxations for the $n-1$ layers before it) Also offers sigificant speedup