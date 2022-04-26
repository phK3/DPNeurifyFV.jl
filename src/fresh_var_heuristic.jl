
# TODO: remove HeuristicScore types and only include the types we actually used?

struct FreshVarHeuristic end

struct HeuristicScore
    # represents utilization factor for symbolic lower bounds in the layers
    lowers::Vector{Float64}
    # represents utilization factor for symbolic upper bounds in the layers
    uppers::Vector{Float64}
end

function backward_linear(solver::FreshVarHeuristic, L::Layer, input::HeuristicScore)
    W⁺ = max.(0., L.weights')
    W⁻ = abs.(min.(0., L.weights'))
    lowers = W⁺ * input.lowers + W⁻ * input.uppers
    uppers = W⁺ * input.uppers + W⁻ * input.lowers
    return HeuristicScore(lowers, uppers)
end

function backward_act(solver::FreshVarHeuristic, L::Layer{ReLU}, input, lbs, ubs)
    λ_l = relaxed_relu_gradient_lower.(lbs, ubs)
    λ_u = relaxed_relu_gradient.(lbs, ubs)

    lowers = λ_l .* input.lowers
    uppers = λ_u .* input.uppers
    return HeuristicScore(lowers, uppers)
end

function backward_act(solver::FreshVarHeuristic, L::Layer{Id}, input, lbs, ubs)
    return input
end

"""
Calculates factors for usage of node's lower and upper bound later on in the network.

input (HeuristicScore) - lower and upper bound usage factors of the output nodes

pre_activation (bool) - whether to return factors for the pre-activation values of a neuron
post_activation (bool) - whether to return factors for the post-activation values of a neuron
"""
function backward_network(solver::FreshVarHeuristic, net, lbs, ubs, input; pre_activation=true, post_activation=true)
    scores = []

    Z = backward_linear(solver, net.layers[end], input)
    post_activation && push!(scores, Z)

    for i in reverse(2:length(net.layers)-1)
        layer = net.layers[i]
        Ẑ = backward_act(solver, layer, Z, lbs[i], ubs[i])
        Z = backward_linear(solver, layer, Ẑ)

        pre_activation && push!(scores, Ẑ)
        post_activation && push!(scores, Z)
    end

    # we iterated backwards through the NN, but want scores forward
    return reverse(scores)
end


function cancellation_score(l, u)
    M = max(l, u)
    m = min(l, u)

    return M == 0. ? 0. : m / M
end

# Methods for calculating amount of overapproximation of nodes

abstract type OverapproxCalculator end

struct AreaCalculator end
struct RangeCalculator end
struct OneSideBound end
struct SplitAreaCalculator end


function calc_overapprox(method::AreaCalculator, l, u)
    l >= 0 && return 0.
    u <= 0 && return 0.
    upper_area = 0.5*(-l)*u
    if u > -l
        lower_area = 0.5*l*l
    else
        lower_area = 0.5*u*u
    end

    return upper_area + lower_area
end


function calc_overapprox(method::SplitAreaCalculator, l, u)
    l >= 0 && return 0., 0.
    u <= 0 && return 0., 0.
    upper_area = 0.5*(-l)*u
    if u > -l
        lower_area = 0.5*l*l
    else
        lower_area = 0.5*u*u
    end

    return upper_area, lower_area
end


function calc_overapprox(method::RangeCalculator, l, u)
    l >= 0 && return 0
    u <= 0 && return 0
    return u - l
end


function calc_overapprox(method::OneSideBound, l, u)
    l >= 0 && return 0
    u <= 0 && return 0
    return min(-l, u)
end


######## Local Layer introduction Heuristics

function get_fresh_var_idxs_largest_range(max_vars, current_n_vars, los, his, var_frac)
    n_node = length(los)
    n_vars = min(max_vars - current_n_vars, floor(Int, var_frac * n_node))
    return fresh_var_largest_range(los, his, n_vars)
end


"""
Returns n indices of entries in lbs, ubs with largest range (ubs[i] - lbs[i])
"""
function fresh_var_largest_range(lbs::Vector{Float64}, ubs::Vector{Float64}, n::Int64)
    n <= 0 && return Int64[]  # early termination, if no variables are introduced

    ranges = ubs - lbs
    p = sortperm(-ranges) # sort descending
    p = p[(lbs[p] .< 0) .& (ubs[p] .> 0)] # want only crossing ReLUs
    return p[1:min(length(p), n)]
end


function fresh_var_first(lbs::Vector{Float64}, ubs::Vector{Float64}, n::Int64)
    crossing_mask = ((lbs .< 0) .& (ubs .> 0))
    crossing = (1:length(lbs))[crossing_mask]
    return crossing[1:min(length(crossing), n)]
end


function fresh_var_earliest(max_vars, current_n_vars, los, his, var_frac)
    n_node = length(los)
    n_vars = min(max_vars - current_n_vars, floor(Int, var_frac * n_node))

    return fresh_var_first(los, his, n_vars)
end


function fresh_var_range_non_zero(max_vars, current_n_vars, los, his, var_frac)
    n_node = length(los)
    n_fixed_zero = sum(his .<= 0)
    n_node = n_node - n_fixed_zero  # fixed zero nodes are basically non-existent in the nn
    n_vars = min(max_vars - current_n_vars, ceil(Int, var_frac * n_node))
    return fresh_var_largest_range(los, his, n_vars)
end


function fresh_var_n_vars_non_zero(max_vars, current_n_vars, los, his, var_frac)
    n_node = length(los)
    n_fixed_zero = sum(his .<= 0)
    n_node = n_node - n_fixed_zero  # fixed zero nodes are basically non-existent in the nn

    λ = n_node / (current_n_vars + n_node)
    n_vars = min(max_vars - current_n_vars, ceil(Int, var_frac * λ * n_node))

    ranges = his - los

    p = sortperm(-ranges)
    p = p[(los[p] .< 0) .& (his[p] .> 0)] # only crossing relus
    return p[1:min(length(p), n_vars)]
end
