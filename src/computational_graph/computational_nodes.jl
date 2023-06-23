

struct Linear{S,N,M,B} <: Node{S,N} where {M<:AbstractMatrix{N},B<:AbstractVector{N}}
    # we really need M,B<:Abstract... instead of M::AbstractMatrix{N} since in Julia
    # Struct{Float64} is no subtype of Struct{Real} for performance reasons
    parents::AbstractVector{S}
    children::AbstractVector{S}
    name::S
    # weights of the layer
    dense::Dense{typeof(identity),M,B}
    # positive weights
    dense⁺::Dense{typeof(identity),M,B}
    # negative weights
    dense⁻::Dense{typeof(identity),M,B}
end


function Linear(parents::Vector{S}, children::Vector{S}, name::S, W::AbstractMatrix{N}, b::AbstractVector{N}; double_precision=false) where {S,N<:Number}
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

    return Linear{S,N,typeof(dense.weight),typeof(dense.bias)}(parents, children, name, dense, dense⁺, dense⁻)
end


function forward_node(L::Linear{S,N}, x) where {S,N}
    return L.dense(x)
end
    


struct Concat{S,N} <: Node{S,N}
    parents::AbstractVector{S}
    children::AbstractVector{S}
    name::S
    dim::Integer
end
