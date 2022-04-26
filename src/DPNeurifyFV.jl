module DPNeurifyFV

using LazySets, NeuralVerification, Parameters, LinearAlgebra, DataStructures, NeuralPriorityOptimizer
using NeuralVerification: TOL, Layer, Network, AbstractNetwork, ActivationFunction, ReLU, Id, n_nodes, relaxed_relu_gradient, compute_output
import NeuralVerification: affine_map, interval_map
import NeuralPriorityOptimizer: split_hyperrectangle, split_largest_interval
const NV = NeuralVerification

# redefinitions of function defined in NeuralVerification.jl
include("overwrite_neural_verification.jl")
include("network_neg_pos_idx.jl")
include("util.jl")
include("symbolic_interval_fv_heur.jl")
include("fresh_var_heuristic.jl")
include("dp_neurify_fv.jl")
include("optimization_bab.jl")

export 
    NetworkNegPosIdx,
    DPNFV,
    init_symbolic_interval_fvheur,
    optimize_linear_deep_poly,
    contained_within_polytope_deep_poly,
    reaches_polytope_deep_poly
    

end # module
