
using DPNeurifyFV, LazySets, NeuralVerification
import VNNLib.NNLoader as NNL

const NV = NeuralVerification
const DP = DPNeurifyFV

onnx_path = "../va-smart/verification/simple_lstm_10_no_initial_state.onnx"
nn = NNL.load_network_dict(DP.CGType, onnx_path)

n_steps = 10
y = DP.propagate(nn, zeros(2, n_steps, 1), verbosity=2);

lb = -0.05 .* ones(2, n_steps, 1)
ub = 0.05 .* ones(2, n_steps, 1)
input_set = Hyperrectangle(low=vec(lb), high=vec(ub))
sz = DP.SplitZonotope(input_set, (2, n_steps, 1));

ẑ = DP.propagate(DP.LSTMSolver(), nn, sz, verbosity=2)