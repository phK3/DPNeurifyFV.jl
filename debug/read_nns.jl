
# test reading onnx files from vnncomp
# and check if concrete execution matches with onnx execution

using DPNeurifyFV, LazySets, PyVnnlib, NeuralVerification, LinearAlgebra, Flux
import VNNLib.NNLoader as NNL
import ONNXRunTime as OX

const NV = NeuralVerification
const DP = DPNeurifyFV


reversedims(A::AbstractArray) = permutedims(A, reverse(tuple(1:ndims(A)...)))


function check_concrete_pass(model_paths; verbosity=0)
    errs = []
    failed_to_load = []
    loaded = []
    for mp in model_paths
        verbosity > 0 && println("loading network ", mp)
        nn = try 
            NNL.load_network_dict(DP.CGType, mp)
        catch e
            println("network at ", mp, " could not be loaded!")
            push!(failed_to_load, mp)
            continue
        end
        in_shape = map(s -> ifelse(typeof(s) <: Integer, s, 1), nn.input_shape)
        y_cg = DP.propagate(nn, zeros(in_shape), verbosity=verbosity)

        model = OX.load_inference(mp)
        in_name = DP.get_inputs(nn.in_node)[1]
        outs = model(Dict(in_name => zeros(Float32, reverse(in_shape))))
        y_flux = outs[DP.get_outputs(nn.out_node)[1]]
        y_flux = reversedims(y_flux)

        if y_cg != y_flux
            push!(errs, (mp, y_cg, y_flux))
        end

        push!(loaded, mp)
        GC.gc()
    end

    println("loaded nns: ", loaded)
    println("failed to load: ", failed_to_load)
    return errs
end


## safeNLP
# both networks parse successfully

onnx_path = "../vnncomp2024_benchmarks/safeNLP/onnx/medical/perturbations_0.onnx"
nn = NNL.load_network_dict(DP.CGType, onnx_path)

# execution works
check_concrete_pass([onnx_path])

# execution works
onnx_path = "../vnncomp2024_benchmarks/safeNLP/onnx/ruarobot/perturbations_0.onnx"
check_concrete_pass([onnx_path])


## nn4sys
# lindex parses successfully
# the other networks don't, mscn and pensieve are too complicated, we could maybe support pensieve_simple ...

onnx_dir = "../vnncomp2023_benchmarks/benchmarks/nn4sys/onnx"
model_paths = filter(endswith(".onnx"), readdir(onnx_dir, join=true))
check_concrete_pass(model_paths, verbosity=2)


## linearizeNN
# the NN can be parsed

onnx_dir = "../vnncomp2024_benchmarks/LinearizeNN_benchmark2024/onnx"
model_paths = filter(endswith(".onnx"), readdir(onnx_dir, join=true))
check_concrete_pass(model_paths)

## distshift
# the NN can be parsed

onnx_dir = "../vnncomp2024_benchmarks/dist-shift/onnx"
model_paths = filter(endswith(".onnx"), readdir(onnx_dir, join=true))
check_concrete_pass(model_paths)

## acasxu

onnx_dir = "../vnncomp2023_benchmarks/benchmarks/acasxu/onnx"
model_paths = filter(endswith(".onnx"), readdir(onnx_dir, join=true))
check_concrete_pass(model_paths)

## cgan
# failed to load: nonlinear activations, upsample
# errors in execution of: other networks!!!

onnx_dir = "../vnncomp2023_benchmarks/benchmarks/cgan/onnx"
model_paths = filter(endswith(".onnx"), readdir(onnx_dir, join=true))
check_concrete_pass(model_paths)


## tllverifybench
# all NNs can be loaded

onnx_dir = "../vnncomp2023_benchmarks/benchmarks/tllverifybench/onnx"
model_paths = filter(endswith(".onnx"), readdir(onnx_dir, join=true))
check_concrete_pass(model_paths)

## vggnet16
# can't load them!!!

onnx_dir = "../vnncomp2023_benchmarks/benchmarks/vggnet16/onnx"
model_paths = filter(endswith(".onnx"), readdir(onnx_dir, join=true))
check_concrete_pass(model_paths)


