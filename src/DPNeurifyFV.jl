module DPNeurifyFV

using LazySets, NeuralVerification, Parameters, LinearAlgebra
import NeuralVerification: Layer, Network, AbstractNetwork, ActivationFunction, ReLU, Id, n_nodes, affine_map, interval_map, relaxed_relu_gradient
const NV = NeuralVerification

# redefinitions of function defined in NeuralVerification.jl
include("overwrite_neural_verification.jl")
include("network_neg_pos_idx.jl")
include("util.jl")
include("symbolic_interval_fv_heur.jl")
include("fresh_var_heuristic.jl")
include("dp_neurify_fv.jl")

export 
    NetworkNegPosIdx,
    DPNFV,
    init_symbolic_interval_fvheur
    

end # module
