
### General

function unit_vec(i, n)
    eᵢ = zeros(n)
    eᵢ[i] = 1.
    return eᵢ
end


function Base.split(H::AbstractHyperrectangle, index::Int)
    lbs, ubs = low(H), high(H)
    split_val = 0.5 * (lbs[index] + ubs[index])

    high1 = copy(ubs)
    high1[index] = split_val
    low2 = copy(lbs)
    low2[index] = split_val

    domain1 = Hyperrectangle(low=lbs, high=high1)
    domain2 = Hyperrectangle(low=low2, high=ubs)

    return domain1, domain2
end


"""
We say that the domain of a hyperrectangle is just itself.

Debatable, if the hyperrectangle represents some interval that originated from an input hyperrectangle.
But we just need it for print_progress in our BaB procedure. (but still might be better not to export it)
"""
domain(h::AbstractHyperrectangle) = h 

### Related to symbolic intervals with fresh variables ###

"""
Substitutes variables in symbolic bounds sym_lo and sym_hi with symbolic lower 
and upper bounds var_los and var_his.

sym_lo - symbolic lower bounds
sym_hi - symbolic upper bounds
var_los - symbolic lower bounds of variables to substitute
var_his - symbolic upper bounds of variables to substitute 
n_in - number of input variables (these cannot be substituted)
n_vars - number of variables 
"""
function substitute_variables(sym_lo, sym_hi, var_los, var_his, n_in, n_vars)
    # for sym_lo
    # should we change the mask to [:, n_in + 1: n_in + n_vars] ???
    var_terms = sym_lo[:, n_in + 1: end - 1]

    var_terms⁺ = max.(var_terms, 0)
    var_terms⁻ = min.(var_terms, 0)

    subs_lb = var_terms⁺ * var_los[1:n_vars, :] .+ var_terms⁻ * var_his[1:n_vars, :] .+ sym_lo[:, (1:n_in) ∪ [end]]

    # for sym_hi
    var_terms .= sym_hi[:, n_in + 1: end - 1]

    var_terms⁺ .= max.(var_terms, 0)
    var_terms⁻ .= min.(var_terms, 0)

    subs_ub = var_terms⁺ * var_his[1:n_vars, :] .+ var_terms⁻ * var_los[1:n_vars, :] .+ sym_hi[:, (1:n_in) ∪ [end]]

    return subs_lb, subs_ub
end


"""
Calculates an input within the hyperrectangle [lbs, ubs] that maximizes the linear
symbolic equation sym_eq.
"""
function maximizer(sym_eq, lbs, ubs)
    if size(sym_eq, 1) == 1
        W⁺ = (sym_eq[1:end-1] .> 0)
        W⁻ = (sym_eq[1:end-1] .< 0)
        maximizer = W⁺ .* ubs + W⁻ .* lbs
    else
        W⁺ = (sym_eq[:, 1:end-1] .> 0)
        W⁻ = (sym_eq[:, 1:end-1] .< 0)
        maximizer =( W⁺' .* ubs + W⁻' .* lbs)'
    end

    return maximizer
end

function minimizer(sym_eq, lbs, ubs)
    # minimizer of eq is just maximizer of -eq
    return maximizer(-sym_eq, lbs, ubs)
end


### related to ReLU state


function is_crossing(lb::Float64, ub::Float64)
    lb < 0 && ub > 0 && return true
    return false
end


function relaxed_relu_gradient_lower(l::Real, u::Real)
    ((u <= 0) || ((l < 0) && (u <= -l))) && return 0.
    return 1.
end

""" Old implementation, ~4x slower
function relaxed_relu_gradient_lower(l::Real, u::Real)
    u <= 0 && return 0.
    l >= 0 && return 1.
    u <= -l && return 0
    return 1.
end
"""


### Network construction

function merge_into_network(network::Network, coeffs::Vector{N} where N<:Number)
    layers = []
    for l in network.layers[1:end-1]
        Ŵ = copy(l.weights)
        b̂ = copy(l.bias)
        σ = l.activation
        push!(layers, Layer(Ŵ, b̂, σ))
    end

    l = network.layers[end]
    if typeof(l.activation) == Id
        Ŵ = coeffs' * l.weights
        b̂ = coeffs' * l.bias
        push!(layers, Layer(Array(Ŵ), [b̂], Id()))
    elseif typeof(l.activation) == ReLU
        Ŵ = copy(l.weights)
        b̂ = copy(l.bias)
        push!(layers, Layer(Ŵ, b̂, ReLU()))
        # just add an additional layer with the linear objective
        push!(layers, Layer(Array(coeffs'), zeros(1), Id()))
    else
        throw(DomainError(l.activation, ": the activation function is not supported!"))
    end

    return Network(layers)
end


# TODO: merge upper function for vectors coeffs into this function
function merge_into_network(network::Network, A::Matrix{N}, b::Vector{N}) where N<:Number
    layers = []
    for l in network.layers[1:end-1]
        Ŵ = copy(l.weights)
        b̂ = copy(l.bias)
        σ = l.activation
        push!(layers, Layer(Ŵ, b̂, σ))
    end

    l = network.layers[end]
    if typeof(l.activation) == Id
        Ŵ = A * l.weights
        # use -b, as we want everything to be satisfied, if output ≤ 0 (Ax ≤ b <-> Ax - b ≤ 0)
        b̂ = A * l.bias - b
        push!(layers, Layer(Ŵ, b̂, Id()))
    elseif typeof(l.activation) == ReLU
        Ŵ = copy(l.weights)
        b̂ = copy(l.bias)
        push!(layers, Layer(Ŵ, b̂, ReLU()))
        # just add an additional layer with the linear objective
        push!(layers, Layer(A, -b, Id()))
    else
        throw(DomainError(l.activation, ": the activation function is not supported!"))
    end

    return Network(layers)
end


### ACAS Xu benchmark

# modified from ZoPE implementation https://github.com/sisl/NeuralPriorityOptimizer.jl (just changed PolytopeComplement to Complement in properties 2-4)
"""
    get_acas_sets(property_number)

Get the input and output sets for acas under the standard definition of a problem 
    as trying to show x in X implies y in Y. This returns the input and output sets X, Y.
    Taken from https://github.com/NeuralNetworkVerification/Marabou/tree/master/resources/properties

"""
function get_acas_sets(property_number)
    if property_number == 1
        input_set = Hyperrectangle(low=[0.6, -0.5, -0.5, 0.45, -0.5], high=[0.6798577687, 0.5, 0.5, 0.5, -0.45])
        output_set = HalfSpace([1.0, 0.0, 0.0, 0.0, 0.0], 3.9911256459)
    elseif property_number == 2
        input_set = Hyperrectangle(low=[0.6, -0.5, -0.5, 0.45, -0.5], high=[0.6798577687, 0.5, 0.5, 0.5, -0.45])
        output_set = Complement(HPolytope([-1.0 1.0 0.0 0.0 0.0; -1.0 0.0 1.0 0.0 0.0; -1.0 0.0 0.0 1.0 0.0; -1.0 0.0 0.0 0.0 1.0], [0.0; 0.0; 0.0; 0.0]))
    elseif property_number == 3
        input_set = Hyperrectangle(low=[-0.3035311561, -0.0095492966, 0.4933803236, 0.3, 0.3], high=[-0.2985528119, 0.0095492966, 0.5, 0.5, 0.5])
        output_set = Complement(HPolytope([1.0 -1.0 0.0 0.0 0.0; 1.0 0.0 -1.0 0.0 0.0; 1.0 0.0 0.0 -1.0 0.0; 1.0 0.0 0.0 0.0 -1.0], [0.0; 0.0; 0.0; 0.0]))
    elseif property_number == 4
        input_set = Hyperrectangle(low=[-0.3035311561, -0.0095492966, 0.0, 0.3181818182, 0.0833333333], high=[-0.2985528119, 0.0095492966, 0.0, 0.5, 0.1666666667])
        output_set = Complement(HPolytope([1.0 -1.0 0.0 0.0 0.0; 1.0 0.0 -1.0 0.0 0.0; 1.0 0.0 0.0 -1.0 0.0; 1.0 0.0 0.0 0.0 -1.0], [0.0; 0.0; 0.0; 0.0]))
    else
        @assert false "Unsupported property number"
    end

    return input_set, output_set
end
