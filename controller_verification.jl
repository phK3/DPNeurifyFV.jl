
using DPNeurifyFV, LazySets, PyVnnlib, NeuralVerification, JuMP, 
      LinearAlgebra, Flux, Gurobi
import VNNLib.NNLoader as NNL

const NV = NeuralVerification
const DP = DPNeurifyFV


onnx_path   = "../underwater/lstm_no_initial_state-sim.onnx"
nn = NNL.load_network_dict(DP.CGType, onnx_path)


# change from gather batch-dim to gather seq-dim
gather_node = DP.Gather(["onnx::Gather_103"], ["onnx::Gemm_106"], "Gather_30", fill(-1), 1)
nn.nodes["Gather_30"] = gather_node
for (k, v) in nn.out_dict
      if v.name == "Gather_30"
            nn.out_dict[k] = gather_node
      end
end


# verification works on logits, not on softmax output, so remove last layer
nn_logits = DP.CompGraph(nn.nodes, nn.in_node, DP.get_producer(nn, "input.8"), nn.out_dict, nn.input_shape, nn.output_shape)


# define input set
lb = -0.05 .* ones(150)
ub =  0.05 .* ones(150)
input_set = Hyperrectangle(low = lb, high = ub)
sz = DP.SplitZonotope(input_set, nn_logits.input_shape)


# propagate through the network
ẑ = DP.propagate(DP.LSTMSolver(), nn_logits, sz)


# optimization
out_spec = HPolytope([1. 0 0 0 0;], [5.])  # test if first output is always less equal 5

params = DP.PriorityOptimizerParameters(max_steps=100, print_frequency=1, stop_frequency=1, verbosity=2, timeout=300)
solver = DP.LSTMSolver()
split_method = sz -> DP.split_split_zonotope(sz, nn_logits.input_shape)

DP.contained_within_polytope_sz(nn_logits, input_set, out_spec, params, split=split_method, solver=solver)