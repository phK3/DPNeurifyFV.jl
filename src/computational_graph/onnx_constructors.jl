

"""
Mirrors weights tensor along each dimension.

Flux convolution is true convolution, so we have to flip the weights from
onnx convolution, which is CrossCorrelation in reality.
"""
flipweights(w::AbstractArray{T,N}) where {T,N} = w[(size(w, i):-1:1 for i = 1:(N-2))..., :, :]

"""
Converts the argument to a tuple, if it is a one dimensional array, 
or does nothing, if the argument is a single integer.

(Flux convolution allows for integer parameters for padding, dilation, ... If 
the parameter is a single integer, it is the same for all dimensions, if there are different
values per dimension, it has to be passed as a TUPLE, not a array.)
"""
function convert2intOrTuple(v::AbstractArray{<:Integer, 1})
    # TOOD: do we need to revert all of these tuples?
    return tuple(v...)
end

function convert2intOrTuple(v::Integer)
    return v
end



"""
Type for parsing Computational Graph
"""
struct CGType <: NNL.NetworkType end




function NNL.construct_layer_add(::Type{CGType}, name, inputs, outputs, a, b)
    println("node $name: a = $a, b = $b\n")
    return Add(inputs, outputs, name)
end

function NNL.construct_layer_add(::Type{CGType}, name, inputs, outputs, a::Type{NNL.DynamicInput}, b)
    println("node $name: add constant\n")
    return AddConst(inputs, outputs, name, b)
end

function NNL.construct_layer_add(::Type{CGType}, name, inputs, outputs, a, b::Type{NNL.DynamicInput})
    println("node $name: add constant\n")
    return AddConst(inputs, outputs, name, a)
end

function NNL.construct_layer_sub(::Type{CGType}, name, inputs, outputs, a, b)
    println("node $name: a = $a, b = $b\n")
    println("parsing Sub with 2 inputs")
    return Sub(inputs, outputs, name)
end

function NNL.construct_layer_sub(::Type{CGType}, name, inputs, outputs, a::Type{NNL.DynamicInput}, b)
    println("node $name: sub constant\n")
    return AddConst(inputs, outputs, name, .-b)
end

function NNL.construct_layer_sub(::Type{CGType}, name, inputs, outputs, a, b::Type{NNL.DynamicInput})
    # here the variable input is subtracted from the constant, so can't handle it with add
    println("node $name: sub constant\n")
    return SubConst(inputs, outputs, name, a)
end


function NNL.construct_layer_mul(::Type{CGType}, name, inputs, outputs, a::Type{NNL.DynamicInput}, b::Type{NNL.DynamicInput})
    println("node $name: multiply")
    return Mul(inputs, outputs, name)
end

function NNL.construct_layer_div(::Type{CGType}, name, inputs, outputs, a::Type{NNL.DynamicInput}, b::Type{NNL.DynamicInput})
    println("node $name: div")
    return Div(inputs, outputs, name)
end


function NNL.construct_layer_matmul(::Type{CGType}, name, inputs, outputs, weight, x::Type{NNL.DynamicInput})
    println("parsing Matmul with input :-)")
    #println("node $name with params $weight and x is $x")
    #println("weight <: DynamicInput? ", typeof(weight) <: NNL.DynamicInput)
    #println("x <: DynamicInput? ", typeof(x) <: NNL.DynamicInput)

    return Linear(inputs, outputs, name, weight, zero(weight[:,1]))
end

# TODO: is Type{NNL.DynamicInput} what we really want here?
function NNL.construct_layer_matmul(::Type{CGType}, name, inputs, outputs, x::Type{NNL.DynamicInput}, weight)
    println("parsing Matmul with input :-)")
    # TODO: do we have to transpose matrix if it is x * W instead of W*x?
    #println("node $name with params $weight and x is $x")
    return Linear(inputs, outputs, name, weight, zero(weight[:,1]))
end

function NNL.construct_layer_gemm(::Type{CGType}, name, inputs, outputs, A, B, C; alpha=1., beta=1., transA=0, transB=0)
    @assert (transA == 0 && A == NNL.DynamicInput) "General Gemm not supported"
    println("parsing Gemm of $(typeof(A)), $(typeof(B)), $(typeof(C))")
    if transB == 0
        W = alpha .* B
    else
        W = alpha .* B'
    end

    b = beta .* C

    return Linear(inputs, outputs, name, W, b)
end




function NNL.construct_layer_conv(::Type{CGType}, name, inputs, outputs, data, weights, bias;
                                  auto_pad="NOTSET", dilations=nothing, group=1, kernel_shape=nothing, pads=nothing, strides=nothing)
    @assert auto_pad == "NOTSET" "auto_pad currently not supported! (node $name)"
    println("parsing Conv!")

    strides = isnothing(strides) ? 1 : convert2intOrTuple(strides)
    dilations = isnothing(dilations) ? 1 : convert2intOrTuple(dilations)
    pads = isnothing(pads) ? 0 : convert2intOrTuple(pads)

    # onnx really calculates CrossCorrelation, so need to flip weights for convolution
    weights = flipweights(weights)

    ret = Convolution(inputs, outputs, name, weights, bias, stride=strides, pad=pads, dilation=dilations, groups=group)
    return ret
end

function NNL.construct_layer_conv_transpose(::Type{CGType}, name, inputs, outputs, data, weights, bias;
                                            auto_pad="NOTSET", dilations=nothing, group=1, kernel_shape=nothing, output_padding=nothing, output_shape=nothing, pads=nothing, strides=nothing)
    println("constructing convT!")
    @assert auto_pad == "NOTSET" "auto_pad currently not supported! (node $name)"
    @assert data == NNL.DynamicInput "Expected DynamicInput for data, but got $data"
    strides = isnothing(strides) ? 1 : convert2intOrTuple(strides)
    dilations = isnothing(dilations) ? 1 : convert2intOrTuple(dilations)
    pads = isnothing(pads) ? 0 : convert2intOrTuple(pads)
    
    !isnothing(output_padding) && @warn("Flux doesn't support output padding, got output_padding = $output_padding")
    !isnothing(output_shape) && @warn("Flux doesn't support output shape, got output_shape = $output_shape")

    weights = flipweights(weights)

    return ConvolutionTranspose(inputs, outputs, name, weights, bias, stride=strides, pad=pads, dilation=dilations, groups=group)
end


function NNL.construct_layer_relu(::Type{CGType}, name, inputs, outputs, data)
    println("parsing ReLU :-)")
    return Relu(inputs, outputs, name)
end

function NNL.construct_layer_sigmoid(::Type{CGType}, name, inputs, outputs, data)
    println("parsing Sigmoid ;-)")
    return Sigmoid(inputs, outputs, name)
end

function NNL.construct_layer_tanh(::Type{CGType}, name, inputs, outputs, data)
    println("parsing Tahn ;-)")
    return Tanh(inputs, outputs, name)
end

function NNL.construct_layer_reducesum(::Type{CGType}, name, inputs, outputs, data; axes=nothing, keepdims=1, noop_with_empty_axes=0)
    println("parsing ReduceSum ;-/")
    return ReduceSum(name, inputs, outputs, axes, keepdims == 1)
end
    

function NNL.construct_layer_flatten(::Type{CGType}, name, inputs, outputs, data; axis=1)
    println("parsing flatten :-)")
    return Flatten(inputs, outputs, name)
end


function NNL.construct_layer_reshape(::Type{CGType}, name, inputs, outputs, data, shape)
    @assert data == NNL.DynamicInput
    println("parsing reshape: $shape")

    # Flux needs WHCN instead of NCHW -> reverse
    # Julia needs : for calculate dim instead of -1 for python
    shape = reverse(tuple(map(x -> ifelse(x > 0, x, :), shape)...))
    
    return Reshape(inputs, outputs, name, shape)
end


function NNL.construct_layer_upsample(::Type{CGType}, name, inputs, outputs, data, scales; mode="nearest")
    @assert data == NNL.DynamicInput "Expected DynamicInput for data, but got $data"
    # NCHW -> WHCN
    scales = reverse(tuple(Integer.(scales)...))
    return Upsampling(inputs, outputs, name, scale=scales)
end


function NNL.construct_layer_slice(::Type{CGType}, name, inputs, outputs, data, starts, ends, axes, steps)
    println("parsing slice")
    return Slice(inputs, outputs, name, starts, ends, axes, steps=steps)
end


function NNL.construct_layer_batch_normalization(::Type{CGType}, name, inputs, outputs, X, scale, B, input_mean, input_var; 
                                                 epsilon=1e-5, momentum=0.9, training_mode=0)
    @assert X == NNL.DynamicInput
    println("parsing BatchNormalization")
    return BatchNormalization(inputs, outputs, name, input_mean, scale, B, input_var, ϵ=epsilon)
end


function NNL.construct_layer_transpose(::Type{CGType}, name, inputs, outputs, data; perm=nothing)
    @assert data == NNL.DynamicInput
    println("parsing Transpose: $perm")

    # Flux needs WHCN instead of NCHW -> reverse
    # Since dims are reversed, the index of the smallest dim needs to be largest and the index of the largest dim needs to be 1 (since Julia is 1-indexed)
    #   -> subract the current index from (max(perm) + 1)
    perm = Tuple(reverse((maximum(perm) + 1) .- perm))
    
    return Transpose(inputs, outputs, name, perm)
end


function NNL.construct_layer_average_pool(::Type{CGType}, name, inputs, outputs, data; auto_pad="NOTSET", 
                                          ceil_mode=0, count_include_pad=0, dilations=nothing, kernel_shape=nothing, 
                                          pads=nothing, strides=nothing)
    @assert auto_pad == "NOTSET" "auto_pad currently not supported! (node $name)"
    @assert ceil_mode == 0 "only ceil_mode = 0 supported! (node $name)"
    @assert count_include_pad == 0 "only count_include_pad = 0 supported! (node $name)"
    @assert isnothing(dilations) "dilations not supported! (node $name)"
    println("parsing AveragePool!")

    strides = isnothing(strides) ? 1 : convert2intOrTuple(strides)
    dilations = isnothing(dilations) ? 1 : convert2intOrTuple(dilations)
    pads = isnothing(pads) ? 0 : convert2intOrTuple(pads)
    window = reverse(Tuple(kernel_shape))

    return AveragePool(inputs, outputs, name, window, stride=strides, pad=pads)   
end


function NNL.construct_layer_squeeze(::Type{CGType}, name, inputs, outputs, data, axes)
    @assert data == NNL.DynamicInput
    println("parsing Squeeze: axes = $axes")
    # Flux needs reversed dimensions, but ONNX Squeeze only stores the axes to be squeezed.
    # so we don't know how many axes there are, which makes it difficult to calculate the right indices.
    # We therefore subtract the respective index from the length of the ndims of the input array, which is known at runtime.
    # Since we subtract from the end, we also don't need to add 1 despite Julia being 1-indexed
    return Squeeze(inputs, outputs, name, axes)
end


function NNL.construct_layer_gather(::Type{CGType}, name, inputs, outputs, data, indices; axis=0)
    println("parsing Gather")
    return Gather(inputs, outputs, name, indices, axis) 
end

function NNL.construct_layer_softmax(::Type{CGType}, name, inputs, outputs, data; axis=-1)
    println("parsing Softmax")
    return Softmax(inputs, outputs, name, axis)
end
    

function NNL.construct_network(::Type{CGType}, inputs, outputs, nodes, input_shape, output_shape)
    println("Constructing the whole NN -- yay -- :-)")
    println("inputs: ", inputs)
    println("outputs: ", outputs)
    @assert length(inputs) == 1 "currently only a unique input node is supported"
    @assert length(outputs) == 1 "currently only a unique output node is supported"

    # println(nodes)

    input_node = nothing
    output_node = nothing
    for (key, value) in nodes
        if inputs[1] in value.inputs
            input_node = value
        end

        if outputs[1] in value.outputs
            output_node = value
        end
    end

    # TODO: really include shapes
    println("input_shape = $input_shape, output_shape = $output_shape")

    # need to change from onnx NCHW to WHCN order of Flux
    input_shape = reverse(tuple(input_shape...))
    output_shape = reverse(tuple(output_shape...))

    return CompGraph(collect(values(nodes)), input_node, output_node, input_shape, output_shape)
end