

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
    return reverse(Tuple(v))
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



"""
Padding in ONNX (that is already reverted) is defined as (pad_1_begin, pad_2_begin, ..., pad_1_end, pad_2_end, ...),
but Flux needs it in the form of (pad_1_begin, pad_1_end, pad_2_begin, pad_2_end, ...)
"""
function convert_onnx_pad(pad::NTuple{N, <:Integer}) where N   
    half = N ÷ 2
    return Tuple([ifelse(iseven(i), pad[half + (i ÷ 2)], pad[(i + 1) ÷ 2]) for i in 1:length(pad)])
end

function NNL.construct_layer_conv(::Type{CGType}, name, inputs, outputs, data, weights, bias;
                                  auto_pad="NOTSET", dilations=nothing, group=1, kernel_shape=nothing, pads=nothing, strides=nothing)
    @assert auto_pad == "NOTSET" "auto_pad currently not supported! (node $name)"
    println("parsing Conv!")

    strides = isnothing(strides) ? 1 : convert2intOrTuple(strides)
    dilations = isnothing(dilations) ? 1 : convert2intOrTuple(dilations)
    pads = isnothing(pads) ? 0 : convert_onnx_pad(convert2intOrTuple(pads))

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
    throw("Sigmoid not implemented!!!")
    # return Sigmoid(inputs, outputs, name)
end

function NNL.construct_layer_tanh(::Type{CGType}, name, inputs, outputs, data)
    println("parsing Tahn ;-)")
    throw("Tanh not implemented!!!")
    r# eturn Tanh(inputs, outputs, name)
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
    @assert count_include_pad == 0 || all(pads .== 0) "only count_include_pad = 0 supported! (node $name) (exception, when pads = 0 in every entry)"
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


function NNL.construct_layer_lstm(::Type{CGType}, name, inputs, outputs, data, W_ih::AbstractArray{<:M}, W_hh::AbstractArray{<:M}, bias=nothing, sequence_lens=nothing, 
                                  initial_h=nothing, initial_c=nothing, P=nothing; activation_alpha=nothing, activation_beta=nothing, activations=nothing, clip=nothing,
                                  direction="forward", hidden_size=-1, input_forget=0, layout=0) where M<:Number
    @assert data == NNL.DynamicInput "Expected DynamicInput for data but got $data"
    @assert hidden_size >= 0 "hidden_size must be set!"
    @assert isnothing(sequence_lens) "implementation can't use sequence_lens, got sequence_lens = $(sequence_lens)"
    @assert isnothing(P) "Peephole connections are not supported!"
    @assert isnothing(activation_alpha) && isnothing(activation_beta) && isnothing(activations) "Only standard activations are supported! Got activation_alpha = $(activation_alpha), 
                                                                                                 activation_beta = $(activation_beta), activations = $activations"
    @assert isnothing(clip) "implementation doesn't support clipping!"
    @assert direction == "forward" "reverse or bidirectional not supported! Got $direction"
    @assert input_forget == 0 "Coupling of input and forget gates is not supported!"
    # TODO: what is the supported layout???
    println("parsing LSTM")

    # | Param | ONNX input shape  | required Flux shape |
    # +-------+-------------------+---------------------+
    # | W_ih  |(n_i, 4*n_h, dirs) |  (4*n_h, n_i)       |
    # | W_hh  |(n_h, 4*n_h, dirs) |  (4*n_h, n_h)       |
    # |b      | (2*4*n_h, dirs)   |  (4*n_h)            |
  
    input_size = size(W_ih, 1)
    hidden_size = size(W_hh, 1)

    # need to reorder columns of W_ih and W_hh
    # Flux has [i f c o] order for input, forget, cell and output gate, while
    # ONNX has [i o f c] order
    reordering = (1:hidden_size) ∪ (2*hidden_size+1:3*hidden_size) ∪ (3*hidden_size+1:4*hidden_size) ∪ (hidden_size+1:2*hidden_size)
    
    num_directions = size(W_ih, 3)  # this should be 1 with asserting direction == "forward"
    W_ih = W_ih[:,reordering,1]'  # have no bidirectional in Flux, so assume weights for first direction are the ones for forward
    W_hh = W_hh[:,reordering,1]'  # transpose to get required shape

    # in ONNX biases for ih and hh are stacked, but only one bias is required, which is the result of the addition of both halves
    bias = isnothing(bias) ? zeros(M, 4*hidden_size) : bias[reordering, 1] .+ bias[reordering .+ 4*hidden_size, 1]
    initial_h = isnothing(initial_h) ? zeros(M, hidden_size, 1) : initial_h
    initial_c = isnothing(initial_c) ? zeros(M, hidden_size, 1) : initial_c

    cell = Flux.LSTMCell(W_ih, W_hh, bias, (initial_h, initial_c))

    return LSTMLayer(inputs, outputs, name, cell, num_directions)
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


