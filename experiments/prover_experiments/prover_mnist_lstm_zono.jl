
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


sz = DP.SplitZonotope(input_set, nn.input_shape)
t_zono = @elapsed ẑ_zono = DP.propagate(DP.LSTMSolver(use_zonotope_domain=true), nn, sz)


## test on MNIST
dataset = MNIST(:test)
X, y = dataset[:];

logfile = "./prover_mnist_results_zono.csv"
open(logfile, "w") do f
    println(f, "idx,lower,upper,time")
end

n_test = 100
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
    nn_spec = DP.merge_into_network(nn, A, b);

    s = DP.SplitZonotope(input_set, nn_spec.input_shape);

    solver = DP.LSTMSolver(use_zonotope_domain=true)

    t_zono = @elapsed ẑ_zono = DP.propagate(solver, nn_spec, s)
    lower, upper = shaped_bounds(ẑ_zono)
    upper = maximum(upper)
    println("upper = ", upper)

    open(logfile, "a") do f
        # we don't have concrete lower bound as we don't search for cex
        println(f, i, ", ", -Inf, ", ", upper, ", ", t_zono)
    end
end
