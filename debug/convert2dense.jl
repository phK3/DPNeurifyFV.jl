
# In order to simplify solver development, we want to convert as many theoretically linear/affine layers as possible to 
# actual dense layers, s.t. solver developers can concentrate on nonlinearities

using DPNeurifyFV, VNNLib
const DP = DPNeurifyFV
const NNL = VNNLib.NNLoader

nn = NNL.load_network_dict(DP.CGType, "../vnncomp2023_benchmarks/benchmarks/nn4sys/onnx/pensieve_big_simple.onnx")
nn_lin = DP.cg2dense(nn, verbosity=1)
nn_summ = DP.summarize_linear_layers(nn_lin, verbosity=2)

y = DP.propagate(nn, zeros(nn.input_shape))
y_lin = DP.propagate(nn_lin, zeros(nn_lin.input_shape))
y_summ = DP.propagate(nn_summ, zeros(nn_summ.input_shape))
