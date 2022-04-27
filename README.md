
# DPNeurifyFV

Neural network verification based on input splitting and forward propagation of symbolic intervals with fresh variables.

## Installation

Using the Julia package manager (type `]` in the Julia REPL) type
- `add https://github.com/phK3/NeuralVerification.jl#BuildingBranch` (adds my fork of `NeuralVerification.jl` to the environment)
    - at least use my fork, until [this issue](https://github.com/sisl/NeuralVerification.jl/issues/201) with the installation of `NeuralVerification.jl` is resolved
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