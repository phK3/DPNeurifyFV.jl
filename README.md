
# DPNeurifyFV

Neural network verification based on input splitting and forward propagation of symbolic intervals with fresh variables.

## Installation

Using the Julia package manager (type `]` in the Julia REPL) type
- `add https://github.com/sisl/NeuralVerification.jl#0d9be34` (adds `NeuralVerification.jl` to the environment, but pinned to a specific commit)
    - at least use the pinned version, until [this issue](https://github.com/sisl/NeuralVerification.jl/issues/201) with the installation of `NeuralVerification.jl` is resolved
- `add https://github.com/sisl/NeuralPriorityOptimizer.jl` and follow the installation instructions on [their repo](https://github.com/sisl/NeuralPriorityOptimizer.jl) 
- `add LazySets`

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