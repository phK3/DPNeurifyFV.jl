
# TODO: make AbstractSymbolicIntervalBounds a subtype of LazySet?
struct SymbolicIntervalFVHeur{F<:AbstractPolytope, N<:Number} <: LazySet{N}
    Low::Matrix{N}
    Up::Matrix{N}
    domain::F
    lbs::Vector{Vector{N}}
    ubs::Vector{Vector{N}}
    var_los::Matrix{N}
    var_his::Matrix{N}
    max_vars::Int
    importance::Vector{N}
    n_layers::Int
end


domain(s::SymbolicIntervalFVHeur) = s.domain
# TODO: should we change this definition? It's not really the radius of the
# symbolic interval, but we need the radius of the domain during splitting
# ALSO: Why do we need LazySets.radius(domain(s)) for NeuralPriorityOptimizer to
# work, why is it not enough to just write radius(domain(s)) ?
LazySets.radius(s::SymbolicIntervalFVHeur) = LazySets.radius(domain(s))
LazySets.dim(s::SymbolicIntervalFVHeur) = size(s.Up, 1)

get_n_sym(s::SymbolicIntervalFVHeur) = size(s.Low, 2) - 1  # number of current symbolic vars
get_n_in(s::SymbolicIntervalFVHeur) = dim(domain(s))  # number of input variables
get_n_vars(s::SymbolicIntervalFVHeur) = get_n_sym(s) - get_n_in(s) # current number of symboli fresh vars


# calculate bounds for a matrix of **equations**, i.e. the lower **or** upper bound of a symbolic interval
function bounds(eq, input_set::H) where H <: Hyperrectangle
    W = eq[:, 1:dim(input_set)]
    b = eq[:, end]

    W⁺ = max.(W, 0)
    W⁻ = min.(W, 0)

    lb = W⁺ * low(input_set) .+ W⁻ * high(input_set) .+ b
    ub = W⁺ * high(input_set) .+ W⁻ * low(input_set) .+ b

    return lb, ub
end


function lower_bounds(eq, input_set::H, lbs, ubs) where H <: Hyperrectangle
    low, up = bounds(eq, input_set)

    low = max.(low, lbs)
    low = min.(low, ubs)

    return low
end


function upper_bounds(eq, input_set::H, lbs, ubs) where H <: Hyperrectangle
    low, up = bounds(eq, input_set)

    up = max.(up, lbs)
    up = min.(up, ubs)

    return up
end


"""
Initialize SymbolicIntervalFVHeur from neural network and input set.
"""
function init_symbolic_interval_fvheur(net::NetworkNegPosIdx, input_set::AbstractPolytope{N}; max_vars=10) where N <: Number
    n = dim(input_set)
    Low = [I zeros(N, n)]
    Up = [I zeros(N, n)]

    lbs = [fill(typemin(N), n_nodes(l)) for l in net.layers]
    ubs = [fill(typemax(N), n_nodes(l)) for l in net.layers]

    var_los = zeros(N, max_vars, dim(input_set) + 1)
    var_his = zeros(N, max_vars, dim(input_set) + 1)

    importance = zeros(N, dim(input_set))

    n_layers = length(net.layers)

    return SymbolicIntervalFVHeur(Low, Up, input_set, lbs, ubs, var_los, var_his,
                                    max_vars, importance, n_layers)
end


"""
Initialize SymbolicIntervalFVHeur from previous SymbolicIntervalFVHeur and new input set.
"""
function init_symbolic_interval_fvheur(s::SymbolicIntervalFVHeur, input_set::AbstractPolytope{N}; max_vars=10) where N <: Number
    n = dim(input_set)
    Low = [I zeros(N, n)]
    Up = [I zeros(N, n)]

    lbs = [copy(ll) for ll in s.lbs]
    ubs = [copy(uu) for uu in s.ubs]

    var_los = zeros(N, max_vars, dim(input_set) + 1)
    var_his = zeros(N, max_vars, dim(input_set) + 1)

    importance = zeros(N, dim(input_set))

    return SymbolicIntervalFVHeur(Low, Up, input_set, lbs, ubs, var_los, var_his,
                                    max_vars, importance, s.n_layers)
end


"""
Initialize SymbolicIntervalFVHeur by changing symbolic lower and upper bound of
existing SymbolicIntervalFVHeur.
"""
function init_symbolic_interval_fvheur(s::SymbolicIntervalFVHeur, Low::Matrix{N}, Up::Matrix{N}) where N <: Number
    return SymbolicIntervalFVHeur(Low, Up, domain(s), s.lbs, s.ubs,
                                    s.var_los, s.var_his, s.max_vars, s.importance, s.n_layers)
end


""" 
Directly substitutes all variables in SymbolicIntervalFVHeur s
"""
function substitute_variables(s::SymbolicIntervalFVHeur)
    n_sym = get_n_sym(s)
    n_in = get_n_in(s)
    current_n_vars = get_n_vars(s)
    return substitute_variables(s.Low, s.Up, s.var_los, s.var_his, n_in, current_n_vars)
end


function maximizer(s::SymbolicIntervalFVHeur)
    subs_sym_lo, subs_sym_hi = substitute_variables(s)
    return maximizer(subs_sym_hi, low(domain(s)), high(domain(s)))
end


function minimizer(s::SymbolicIntervalFVHeur)
    subs_sym_lo, subs_sym_hi = substitute_variables(s)
    return minimizer(subs_sym_lo, low(domain(s)), high(domain(s)))
end


"""
Returns number of crossing ReLUs in the network as well as a list
of crossing ReLUs per layer.

lbs, ubs are lower and upper bounds of the intermediate neurons
(no input bounds or output bounds)
"""
function get_num_crossing(lbs, ubs)
    n_layers = length(lbs)
    n_crossings = zeros(Int, n_layers)
    for (i, (l, u)) in enumerate(zip(lbs, ubs))
        n_crossings[i] = sum((l .< 0) .& (u .> 0))
    end

    return sum(n_crossings), n_crossings
end

get_num_crossing(s::SymbolicIntervalFVHeur) = get_num_crossing(s.lbs, s.ubs)


function split_symbolic_interval_fv_heur(s::SymbolicIntervalFVHeur{<:Hyperrectangle}, index::Int)
    domain1, domain2 = split(domain(s), index)

    current_n_vars = get_n_vars(s)
    # can't have more vars than parent node?
    s1 = init_symbolic_interval_fvheur(s, domain1; max_vars=current_n_vars)
    s2 = init_symbolic_interval_fvheur(s, domain2; max_vars=current_n_vars)

    return [s1, s2]
end


"""
Splitting based on intermediate importance scores (i.e. coefficients in all crossing ReLUs)
"""
function split_important_interval(s::SymbolicIntervalFVHeur{<:Hyperrectangle})
    radius = high(domain(s)) - low(domain(s))
    # if there are no more crossing ReLUs, importance will be zero vector -> if we
    # multiply it with radius, all inputs are equally important, and argmax always
    # returns the first index -> infinitely loop splitting
    most_important_dim = sum(s.importance) == 0. ? argmax(radius) : argmax(s.importance .* radius)
    return split_symbolic_interval_fv_heur(s, most_important_dim)
end


"""
Splitting based on intermediate importance scores (i.e. coefficients in all crossing ReLUs) with number of fresh variables to
be introduced following NeuroDiff paper
"""
function split_important_interval_neurodiff(s::SymbolicIntervalFVHeur{<:Hyperrectangle})
    radius = high(domain(s)) - low(domain(s))
    # if there are no more crossing ReLUs, importance will be zero vector -> if we
    # multiply it with radius, all inputs are equally important, and argmax always
    # returns the first index -> infinitely loop splitting
    most_important_dim = sum(s.importance) == 0. ? argmax(radius) : argmax(s.importance .* radius)

    domain1, domain2 = split(domain(s), most_important_dim)

    num_crossing, crossings = get_num_crossing(s)
    # don't count last layer as it has Id activation function
    n_vars = ceil(Integer, sum([ncr/i for (i, ncr) in enumerate(crossings[1:end-1])]))

    s1 = init_symbolic_interval_fvheur(s, domain1; max_vars=n_vars)
    s2 = init_symbolic_interval_fvheur(s, domain2; max_vars=n_vars)

    return [s1, s2]
end