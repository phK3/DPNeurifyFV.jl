
using DPNeurifyFV, LazySets, PyVnnlib, NeuralVerification, JuMP, 
      LinearAlgebra, Flux, Gurobi, Accessors
using MLDatasets: MNIST
import VNNLib.NNLoader as NNL

const NV = NeuralVerification
const DP = DPNeurifyFV


function shaped_bounds(sz::DP.SplitZonotope; skip_lp=true)
    lb = DP.bounds(sz, skip_lp=skip_lp, upper=false)
    ub = DP.bounds(sz, skip_lp=skip_lp, upper=true)

    return reshape(lb, sz.shape), reshape(ub, sz.shape)
end


#onnx_path = "../underwater/lstm_modified_14f_56i_32h_1l.onnx"
onnx_path = "../underwater/lstm_modified_4f_196i_32h_1l.onnx"
nn = NNL.load_network_dict(DP.CGType, onnx_path);

## Test on random input 
# define input set (some arbitrary input set for now)
lb = -0.05 .* ones(784)
ub =  0.05 .* ones(784)
input_set = Hyperrectangle(low=lb, high=ub)
sz = DP.SplitZonotope(input_set, nn.input_shape)

ẑ = DP.propagate(DP.LSTMSolver(), nn, sz)


#sz = DP.SplitZonotope(input_set, nn.input_shape)
#t_zono = @elapsed ẑ_zono = DP.propagate(DP.LSTMSolver(use_zonotope_domain=true), nn, sz)


## test on MNIST
dataset = MNIST(:test)
X, y = dataset[:];

logfile = "./prover_mnist_results.csv"
open(logfile, "w") do f
    println(f, "idx,lower,upper,steps,time")
end

n_test = 1000
ϵ = 0.007
cool_var = nothing
for i in 1:n_test
    println("### ", i, " ###")
    x = Float64.(reshape(X[:,:,i], nn.input_shape))
    y_true = y[i]

    ŷ = DP.propagate(DP.ConcreteExecution(), nn, x)
    
    # need to take care of 1-indexed vs 0-indexed
    if argmax(vec(ŷ)) - 1 != y_true
        println("\twrong prediction for center")
        continue
    end

    lb = vec(x) .- ϵ
    ub = vec(x) .+ ϵ
    input_set = Hyperrectangle(low=lb, high=ub)

    A, b = DP.robustness_output_spec(10, y_true + 1)
    out_spec = HPolytope(A, b)

    params = DP.PriorityOptimizerParameters(max_steps=9999, print_frequency=1, stop_frequency=1, verbosity=2, timeout=60)
    solver = DP.LSTMSolver()
    split_method = sz -> DP.split_split_zonotope_importance(sz, nn.input_shape, lstm_split_method=:optimal)

    t = @elapsed x, lower, upper, steps = DP.contained_within_polytope_sz_lp(nn, input_set, out_spec, params, split=split_method, solver=solver)
    println("lower = ", lower, ", upper = ", upper)

    open(logfile, "a") do f
        println(f, i, ", ", lower, ", ", upper, ", ", steps, ", ", t)
    end
end
