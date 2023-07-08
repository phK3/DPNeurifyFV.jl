using DPNeurifyFV, PyVnnlib, LazySets, Flux, NeuralVerification, NeuralPriorityOptimizer
const DP = DPNeurifyFV

params = DP.PriorityOptimizerParameters(max_steps=3, print_frequency=1, stop_frequency=1, verbosity=2)
solver = DPNFV(method=:DeepPolyRelax)
onnx_file = string(@__DIR__, "/../networks/precompile_nns/ACASXU_run2a_1_1_batch_2000.onnx")
vnnlib_file = string(@__DIR__,"/../networks/precompile_nns/prop_1.vnnlib")
DP.verify_vnnlib(solver, onnx_file, vnnlib_file, params, printing=true);