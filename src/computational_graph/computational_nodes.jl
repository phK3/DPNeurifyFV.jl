

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

    dense⁺.weight .= W
    dense⁺.bias .= b

    # bias is already accounted for in dense⁺
    dense⁻.weight .= W
    dense⁻.bias .= zero(b)

    return Linear(parents, children, name, dense, dense⁺, dense⁻)
end


function forward_node(L::Linear, x)
    return L.dense(x)
end
    


struct Concat <: Node
    parents::AbstractVector
    children::AbstractVector
    name
    dim::Integer
end
