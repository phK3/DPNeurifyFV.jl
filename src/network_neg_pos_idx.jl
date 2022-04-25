

############# Network with [W]⁺, [W]⁻ and layers that store their index ########

struct LayerNegPosIdx{F<:ActivationFunction, N<:Number}
    weights::Matrix{N}
    bias::Vector{N}
    W_neg::Matrix{N}
    W_pos::Matrix{N}
    activation::F
    index::Int64
end

LayerNegPosIdx(L::Layer, idx::Int64) = LayerNegPosIdx(L.weights, L.bias, min.(L.weights, 0), max.(L.weights, 0),
                                                           L.activation, idx)
NV.n_nodes(L::LayerNegPosIdx) = length(L.bias)
NV.affine_map(L::LayerNegPosIdx, x)  = L.weights*x .+ L.bias


struct NetworkNegPosIdx <: AbstractNetwork
    layers::Vector{LayerNegPosIdx}
end

NetworkNegPosIdx(net::Network) = NetworkNegPosIdx([LayerNegPosIdx(L, i) for (i, L) in enumerate(net.layers)])

function NV.compute_output(nnet::NetworkNegPosIdx, input)
    curr_value = input
    for layer in nnet.layers
        curr_value = layer.activation(affine_map(layer, curr_value))
    end
    return curr_value
end


### Network with layers that store positive and negative weights separately (and not their index)


struct LayerNegPos{F<:ActivationFunction, N<:Number}
    weights::Matrix{N}
    bias::Vector{N}
    W_neg::Matrix{N}
    W_pos::Matrix{N}
    activation::F
end


LayerNegPos(L::Layer) = LayerNegPos(L.weights, L.bias, min.(L.weights, 0),
                                    max.(L.weights, 0), L.activation)
Layer(L::LayerNegPos) = Layer(L.weights, L.bias, L.activation)

NV.n_nodes(L::LayerNegPos) = length(L.bias)
NV.affine_map(L::LayerNegPos, x) = L.weights*x + L.bias


struct NetworkNegPos <: AbstractNetwork
    layers::Vector{LayerNegPos}
end


NetworkNegPos(net::Network) = NetworkNegPos([LayerNegPos(l) for l in net.layers])


function NV.compute_output(nnet::NetworkNegPos, input)
    curr_value = input
    for layer in nnet.layers
        curr_value = layer.activation(affine_map(layer, curr_value))
    end
    return curr_value
end


function NV.interval_map(W⁻::AbstractMatrix{N}, W⁺::AbstractMatrix{N},
    l::AbstractVecOrMat, u::AbstractVecOrMat) where N
    l_new = W⁺ * l .+ W⁻ * u
    u_new = W⁺ * u .+ W⁻ * l

    return (l_new, u_new)
end