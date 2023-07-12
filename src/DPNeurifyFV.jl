module DPNeurifyFV

using LazySets, NeuralVerification, Parameters, LinearAlgebra, DataStructures, NeuralPriorityOptimizer, CSV, 
        OnnxReader, VnnlibParser, Flux, VNNLib, PyVnnlib, PrecompileTools
using NeuralVerification: TOL, Layer, Network, AbstractNetwork, ActivationFunction, ReLU, Id, n_nodes, relaxed_relu_gradient, compute_output
import NeuralVerification: affine_map, interval_map
import NeuralPriorityOptimizer: split_hyperrectangle, split_largest_interval, split_multiple_times
import VNNLib.NNLoader
const NV = NeuralVerification
const NPO = NeuralPriorityOptimizer
const NNL = NNLoader

# redefinitions of function defined in NeuralVerification.jl
# include("overwrite_neural_verification.jl")
include("network_neg_pos_idx.jl")
include("util.jl")
include("symbolic_interval_fv_heur.jl")
include("fresh_var_heuristic.jl")
include("dp_neurify_fv.jl")
include("optimization_bab.jl")
include("overwrite_zope.jl")
include("verify_vnnlib.jl")
include("computational_graph/computational_graph.jl")
include("computational_graph/computational_nodes.jl")
include("computational_graph/symbolic_interval_graph.jl")
include("computational_graph/dp_neurify_fv_graph.jl")
include("computational_graph/onnx_constructors.jl")
include("computational_graph/cg_optimization_bab.jl")


include("computational_graph/lstm/LSTMRelaxation.jl")

using .LSTMRelaxation

include("computational_graph/lstm/split_zono.jl")
include("computational_graph/lstm/lstm_solver.jl")

# does it get precompiled?
println("precompiling...")
@setup_workload begin
    params = PriorityOptimizerParameters(max_steps=3, print_frequency=1, stop_frequency=1, verbosity=2)
    solver = DPNFV(method=:DeepPolyRelax)
    onnx_file = string(@__DIR__, "/../networks/precompile_nns/ACASXU_run2a_1_1_batch_2000.onnx")
    vnnlib_file = string(@__DIR__,"/../networks/precompile_nns/prop_1.vnnlib")
    verify_vnnlib(solver, onnx_file, vnnlib_file, params, printing=true);
end


export 
    NetworkNegPosIdx,
    DPNFV,
    init_symbolic_interval_fvheur,
    optimize_linear_deep_poly,
    contained_within_polytope_deep_poly,
    reaches_polytope_deep_poly,
    print_network_structure,
    verify_vnnlib
    #general_priority_optimization
    

end # module
