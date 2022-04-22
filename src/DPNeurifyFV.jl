module DPNeurifyFV

using LazySets, NeuralVerification, Parameters
import NeuralVerification: Layer, Network, ActivationFunction
const NV = NeuralVerification

# redefinitions of function defined in NeuralVerification.jl
include("overwrite_neural_verification.jl")
include("network_neg_pos_idx.jl")
include("util.jl")
include("symbolic_interval_fv_heur.jl")
include("dp_neurify_fv.jl")

export 
    NetworkNegPosIdx,
    DPNFV,
    init_symbolic_interval_fvheur
    

end # module
