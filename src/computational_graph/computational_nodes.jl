

struct Linear <: Node
    # vector of identifiers for each input
    inputs::AbstractVector
    # vector of identifiers for each output
    outputs::AbstractVector
    name
    # weights of the layer
    dense::Dense
    # positive weights
    dense⁺::Dense
    # negative weights
    dense⁻::Dense
end


function Linear(inputs::AbstractVector{S}, outputs::AbstractVector{S}, name::S, W::AbstractMatrix{N}, b::AbstractVector{N}; double_precision=false) where {S,N<:Number}
    n_out, n_in = size(W)

    if double_precision
        dense = Dense(n_in => n_out) |> f64
        dense⁺ = Dense(n_in => n_out) |> f64
        dense⁻ = Dense(n_in => n_out) |> f64
    else
        dense = Dense(n_in => n_out)
        dense⁺ = Dense(n_in => n_out)
        dense⁻ = Dense(n_in => n_out)
    end

    dense.weight .= W
    dense.bias .= b

    dense⁺.weight .= max.(0, W)
    dense⁺.bias .= zero(dense.bias)

    dense⁻.weight .= min.(0, W)
    dense⁻.bias .= zero(dense.bias)

    return Linear(inputs, outputs, name, dense, dense⁺, dense⁻)
end


function forward_node(solver, L::Linear, x)
    return L.dense(x)
end


struct AddConst <: Node 
    inputs::AbstractVector
    outputs::AbstractVector
    name
    c
end

function forward_node(solver, L::AddConst, x)
    return x .+ L.c
end

struct SubConst <: Node 
    inputs::AbstractVector
    outputs::AbstractVector
    name
    c
end

function forward_node(solver, L::SubConst, x)
    return L.c .- x
end


# Need to write it as Relu as ReLU is in NeuralVerification and relu is in Flux
struct Relu <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
end


function forward_node(solver, L::Relu, x)
    return relu(x)
end


struct Sigmoid <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
end


function forward_node(solver, L::Sigmoid, x)
    return Flux.σ.(x)
end 


struct Tanh <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
end


function forward_node(solver, L::Tanh, x)
    return tanh.(x)
end 
    

struct Softmax <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
    axis
end


function forward_node(solver, L::Softmax, x)
    axis = ndims(x) - L.axis  # NCHW -> WHCN
    return Flux.NNlib.softmax(x, dims=axis)
end
    

struct Concat <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
    dim::Integer
end


function forward_node(solver, L::Concat, x₁, x₂)
    return cat(x₁, x₂, dims=L.dim)
end


struct Convolution <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
    conv::Conv
    conv⁺::Conv
    conv⁻::Conv
end


"""
Creates an instance of a Convolutional node.

For Conv((k₁, ..., kₙ), c_in => c_out), we have
- size(weight) = (k₁, ..., kₙ, c_in, c_out)
- size(bias) = (c_out,)

args:
    inputs
    outputs
    name
    weight
    bias

kwargs:
    stride - either an Integer or a TUPLE of integers, representing the stride in each dimension
    pad - either an Integer or a TUPLE of integers, representing the padding in each dimension
    dilation - eiher an Integer or a TUPLE of integers, representing the dilation in each dimension
    groups - Integer representing number of groups
    double_precision - whether to use double precision weights
"""
function Convolution(inputs::AbstractVector{S}, outputs::AbstractVector{S}, name::S, 
                    weight, bias; stride=1, pad=0, dilation=1, groups=1, double_precision=false) where S
    # TODO: is this correct?
    kernel_size = size(weight)[1:end-2]
    in_channels, out_channels = size(weight)[end-1:end]

    conv = Conv(kernel_size, in_channels => out_channels, bias=bias, stride=stride, pad=pad, dilation=dilation, groups=groups)
    conv⁺ = Conv(kernel_size, in_channels => out_channels, bias=false, stride=stride, pad=pad, dilation=dilation, groups=groups)
    conv⁻ = Conv(kernel_size, in_channels => out_channels, bias=false, stride=stride, pad=pad, dilation=dilation, groups=groups)

    if double_precision
        conv = conv |> f64
        conv⁺ = conv⁺ |> f64
        conv⁻ = conv⁻ |> f64
    end

    conv.weight .= weight
    #conv.bias .= bias
 
    conv⁺.weight .= max.(0, weight)
    #conv⁺.bias .= zero(conv.bias)

    conv⁻.weight .= min.(0, weight)
    #conv⁻.bias .= zero(conv.bias)

    return Convolution(inputs, outputs, name, conv, conv⁺, conv⁻)
end


function forward_node(solver, L::Convolution, x)
    return L.conv(x)
end


struct ConvolutionTranspose <: Node 
    inputs::AbstractVector
    outputs::AbstractVector
    name
    convt::ConvTranspose
    convt⁻::ConvTranspose
    convt⁺::ConvTranspose
end


"""
Create a transposed convolution node.

For ConvTranspose((k₁, ..., kₙ), c_in => c_out), we have
- size(weight) = (k₁, ..., kₙ, c_out, c_in)
- size(bias) = (c_out,)

"""
function ConvolutionTranspose(inputs::AbstractVector{S}, outputs::AbstractVector{S}, name::S, 
    weight, bias; stride=1, pad=0, dilation=1, groups=1, double_precision=false) where S
    # TODO: is this correct?
    kernel_size = size(weight)[1:end-2]
    # for ConvT order is swapped when compared to Conv
    out_channels, in_channels = size(weight)[end-1:end]

    convt = ConvTranspose(kernel_size, in_channels => out_channels, bias=bias, stride=stride, pad=pad, dilation=dilation, groups=groups)
    convt⁺ = ConvTranspose(kernel_size, in_channels => out_channels, bias=false, stride=stride, pad=pad, dilation=dilation, groups=groups)
    convt⁻ = ConvTranspose(kernel_size, in_channels => out_channels, bias=false, stride=stride, pad=pad, dilation=dilation, groups=groups)

    if double_precision
        convt = convt |> f64
        convt⁺ = convt⁺ |> f64
        convt⁻ = convt⁻ |> f64
    end

    # if i uncomment the bias lines, convt.bias is all zeros, why is that???
    convt.weight .= weight
    #convt.bias .= bias

    convt⁺.weight .= max.(0, weight)
    #convt⁺.bias .= zero(convt.bias)

    convt⁻.weight .= min.(0, weight)
    #convt⁻.bias .= zero(convt.bias)

    return ConvolutionTranspose(inputs, outputs, name, convt, convt⁻, convt⁺)
end


function forward_node(solver, L::ConvolutionTranspose, x)
    return L.convt(x)
end


struct AveragePool <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
    avg::MeanPool
end


function AveragePool(inputs, outputs, name, window; pad=0, stride=window)
    avg = Flux.MeanPool(window, pad=pad, stride=stride)

    return AveragePool(inputs, outputs, name, avg)
end


function forward_node(solver, L::AveragePool, x)
    return L.avg(x)
end


struct Reshape <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
    shape
end


function forward_node(solver, L::Reshape, x)
    return reshape(x, L.shape)
end


struct Flatten <: Node 
    inputs::AbstractVector
    outputs::AbstractVector
    name
end


function forward_node(solver, L::Flatten, x)
    return reshape(x, :, size(x)[end])
end


struct BatchNormalization <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
    batchnorm::BatchNorm
    batchnorm⁺::BatchNorm
    batchnorm⁻::BatchNorm
end


function BatchNormalization(inputs, outputs, name, μ, γ, β, σ²; λ=identity, ϵ=1e-5, double_precision=false)
    channels = length(γ)

    batchnorm = BatchNorm(channels)
    # β is initialized with 0, same for μ 
    batchnorm⁻ = BatchNorm(channels)
    batchnorm⁺ = BatchNorm(channels)
    bns = [batchnorm, batchnorm⁻, batchnorm⁺]

    if double_precision
        for i in eachindex(bns)
            bns[i] = bns[i] |> f64
        end
    end

    for i in eachindex(bns)
        bns[i].σ² .= σ²
        bns[i].ϵ = ϵ
    end

    batchnorm.μ .= μ
    batchnorm.γ .= γ
    batchnorm.β .= β
    batchnorm.λ = λ

    batchnorm⁻.γ .= min.(0, γ)
    batchnorm⁺.γ .= max.(0, γ)

    # don't update parameters
    testmode!(batchnorm)
    testmode!(batchnorm⁻)
    testmode!(batchnorm⁺)

    return BatchNormalization(inputs, outputs, name, batchnorm, batchnorm⁺, batchnorm⁻)
end

    
function forward_node(solver, L::BatchNormalization, x)
    return L.batchnorm(x)
end


struct Upsampling <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
    upsampling::Upsample
end


function Upsampling(inputs, outputs, name; mode=:nearest, scale=nothing, size=nothing)
    @assert ~isnothing(scale) || ~isnothing(size) "Either size or scale needs to be set! (constructor of $name)"
    upsampling = Upsample(mode, scale=scale, size=size)
    return Upsampling(inputs, outputs, name, upsampling)
end


function forward_node(solver, L::Upsampling, x)
    return L.upsampling(x)
end


struct Add <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
end


function forward_node(solver, L::Add, x₁, x₂)
    return x₁ .+ x₂
end


struct Sub <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
end


function forward_node(solver, L::Sub, x₁, x₂)
    return x₁ .- x₂
end


struct Gather <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
    inds::AbstractArray{<:Integer}
    axis::Integer
end


function Gather(inputs, outputs, name, inds; axis=1)
    return Gather(inputs, outputs, name, inds, axis)
end


function my_gather(x::AbstractArray, inds::Vector{<:Integer}; axis=1)
    axis = ndims(x) - axis  # NCHW -> WHCN
    x = transpose_tensor(x, axis, ndims(x))
    inds = get_positive_index.(inds, size(x, axis))

    g = Flux.NNlib.gather(x, inds)
    g = transpose_tensor(g, axis, ndims(x))

    if lenth(inds) == 1
        # TODO: can we get rid of this special case?
        g = g[:,:,1]
    end

    return g
end 


"""
Gathers subtensors specified by lists of indices in inds along the given axis.

args:
    x - the input tensor
    inds - List of vectors of indices

kwargs:
    axis - along which dimension to gather subtensors

returns:
    a tensor of ndims(x)+1, with the first dimension equal to length(inds)
"""
#=
function my_gather(x, inds; axis=1)
    x = transpose_tensor(x, axis, ndims(x))
    inds = get_positive_index.(inds, size(x, axis))
    
    # inds is a vector of tuples of indices
    gs = []
    for idx in inds
        g = Flux.NNlib.gather(x, idx)
        # transpose back and add dimension to concatenate results
        g = transpose_tensor(g, axis, ndims(x))
        push!(gs, Flux.unsqueeze(g, dims=1))
    end
    
    return cat(gs..., dims=1)
end
=#

"""
Gather elements from the specified axis of x.

args:
    x - tensor to gather from
    inds - vector of tensors/lists of indices

kwargs:
    axis - axis to take from (1 rows, 2 cols, 3 ...)
"""
function my_gather(x::AbstractArray, inds::AbstractVector; axis=1)
    axis = ndims(x) - axis  # NCHW -> WHCN
    inds = get_positive_index.(inds, size(x, axis))
    idxs = [Tuple(ifelse(i == axis, ind, :) for i in 1:ndims(x)) for ind in inds]
    return [x[idx...] for idx in idxs]
end


function my_gather(x::AbstractArray, inds::Array{N, 0}; axis=1) where N<:Number
    # special case for zero-dim arrays (arrays holding just one scalar value)
    # TOOD: can we somehow get rid of that special case???
    axis = ndims(x) - axis  # NCHW -> WHCN
    inds = get_positive_index.(inds, size(x, axis))
    idx = Tuple(ifelse(i == axis, inds[], :) for i in 1:ndims(x))
    return x[idx...]
end


function forward_node(solver, L::Gather, x)
    return my_gather(x, L.inds, axis=L.axis)
end
    

struct Slice <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
    starts::AbstractArray{<:Integer}
    # writing ends in Julia is annoying!
    stops::AbstractArray{<:Integer}
    axes::AbstractArray{<:Integer}
    steps::AbstractArray{<:Integer}
end


function Slice(inputs, outputs, name, starts, stops, axes; steps=1)
    @assert all(starts .>= 0) && all(ends .>= 0) "Negative starts or ends are currently not supported! (@ $(Node.name))"
    return Slice(inputs, outputs, name, starts, stops, axes, steps)
end


"""
Calculates slice of tensor x along the specified axes.

The result is x̂, s.t. x̂[:,...,axis,...,:] = x[axis][starts[axis]:steps[axis]:ends[axis]]

args:
    x - the tensor to slice
    starts - starting indices, s.t. starts[axis] is the starting index for axis
    stops - end indices of the slice, s.t. stops[axis] is the end index for axis
    axes - the axes to slice
    steps - stepsizes, s.t. steps[axis] are the steps for that axis
"""
function my_slice(x::AbstractArray, starts, stops, axes; steps=1)
    starts0 = ones(Integer, ndims(x))
    stops0  = [size(x)...]
    steps0  = ones(Integer, ndims(x))
    
    starts0[axes] .= starts
    stops0[axes] .= stops
    steps0[axes] .= steps
    
    starts0 = clamp.(starts0, zero(starts0),  size(x))
    stops0 = clamp.(stops0, zero(stops0), size(x))
    
    inds = [a:b:c for (a,b,c) in zip(starts0, steps0, stops0)]
    return x[inds...]
end


function forward_node(solver, L::Slice, x)
    return my_slice(x, L.starts, L.stops, L.axes, steps=L.steps)
end


struct SplitNode <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
    axis::Integer
    splits
    num_outputs::Integer
end


function SplitNode(inputs, outputs, name; splits=nothing, num_outputs=nothing, axis=1)
    @assert ~isnothing(splits) || ~isnothing(num_outputs) "Either splits or num_outputs has to be set (@ node $(name))"
    if isnothing(num_outputs)
        num_outputs = length(splits)
    end
    # since we don't know the input dimensions, we can't set splits here 
    return SplitNode(inputs, outputs, name, axis, splits, num_outputs)
end


function forward_node(solver, L::SplitNode, x)
    if isnothing(L.splits)
        # -> num_outputs must be set
        ranges = Iterators.partition(1:size(x, L.axis), ceil(Integer, size(x, L.axis) / L.num_outputs))
    else
        # -> splits must be set
        starts = [0; cumsum(L.splits[1:end-1])]
        stops  = cumsum(L.splits)
        ranges = [start+1:stop for (start, stop) in zip(starts, stops)]
    end

    outputs = []
    for inds in ranges
        idxs = [1:size(x,i) for i in 1:ndims(x)]
        idxs[L.axis] = inds
        push!(outputs, x[idxs...])
    end

    return outputs
end


struct Transpose <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
    perm
end


function forward_node(solver, L::Transpose, x)
    return permutedims(x, L.perm)
end


struct Squeeze <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name
    # if axes == nothing, then all singleton dimensions will be removed
    axes
end


function forward_node(solver, L::Squeeze, x)
    axes = isnothing(L.axes) ? Tuple(findall(size(x) .== 1)) : ndims(x) .- L.axes
    @assert all(size(x)[axes] .== 1) "Can't squeeze non-singleton dimensions! Got size(x) = $(size(x)) for axes $axes !"

    return dropdims(x, dims=Tuple(axes))
end


# TODO: Maybe better to use Flux here?
struct LSTMCell <: Node
    inputs::AbstractVector
    outputs::AbstractVector
    name::String
    linear_ih::Linear
    linear_hh::Linear
    state0::AbstractArray
end

struct LSTMLayer <: Node
    cell::LSTMCell
end


function LSTMCell(inputs, outputs, name, Wih, Whh, b; state0=nothing)
    hs4, n_in = size(Wih)
    hidden_size = floor(Integer, hs4 / 4)

    linear_ih = Linear([], [], name * "_linear_ih", Wih, b)
    linear_hh = Linear([], [], name * "_linear_hh", Whh, zeros(hs4))

    if isnothing(state0)
        state0 = zeros(n_in)
    end

    return LSTMCell(inputs, outputs, name, linear_ih, linear_hh, state0)
end
    