

struct Linear <: Node
    parents::AbstractVector
    children::AbstractVector
    name
    # weights of the layer
    dense::Dense
    # positive weights
    dense⁺::Dense
    # negative weights
    dense⁻::Dense
end


function Linear(parents::AbstractVector{S}, children::AbstractVector{S}, name::S, W::AbstractMatrix{N}, b::AbstractVector{N}; double_precision=false) where {S,N<:Number}
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

    return Linear(parents, children, name, dense, dense⁺, dense⁻)
end


function forward_node(solver, L::Linear, x)
    return L.dense(x)
end


# Need to write it as Relu as ReLU is in NeuralVerification and relu is in Flux
struct Relu <: Node
    parents::AbstractVector
    children::AbstractVector
    name
end


function forward_node(solver, L::Relu, x)
    return relu(x)
end
    

struct Concat <: Node
    parents::AbstractVector
    children::AbstractVector
    name
    dim::Integer
end


function forward_node(solver, L::Concat, x₁, x₂)
    return cat(x₁, x₂, dims=L.dim)
end


struct Convolution <: Node
    parents::AbstractVector
    children::AbstractVector
    name
    conv::Conv
    conv⁺::Conv
    conv⁻::Conv
end


function Convolution(parents::AbstractVector{S}, children::AbstractVector{S}, name::S, 
                    weight, bias; stride=1, pad=0, dilation=1, double_precision=false) where S
    # TODO: is this correct?
    kernel_size = size(weight)[1:end-2]
    in_channels, out_channels = size(weight)[end-2:end]

    conv = Conv(kernel_size, in_channels => out_channels, bias=bias, stride=stride, pad=pad, dilation=dilation)
    conv⁺ = Conv(kernel_size, in_channels => out_channels, bias=bias, stride=stride, pad=pad, dilation=dilation)
    conv⁻ = Conv(kernel_size, in_channels => out_channels, bias=bias, stride=stride, pad=pad, dilation=dilation)

    if double_precision
        conv = conv |> f64
        conv⁺ = conv⁺ |> f64
        conv⁻ = conv⁻ |> f64
    end

    conv.weight = weight
    conv.bias = bias
 
    conv⁺.weight = max.(0, weight)
    conv⁺.bias = bias

    conv⁻.weight = min.(0, weight)
    conv⁻.bias = zero(bias)

    return Convolution(parents, children, name, conv, conv⁺, conv⁻)
end


function forward_node(solver, L::Convolution, x)
    return L.conv(x)
end


struct Reshape <: Node
    parents::AbstractVector
    children::AbstractVector
    name
    shape
end


function forward_node(solver, L::Reshape, x)
    return reshape(x, L.shape)
end




    