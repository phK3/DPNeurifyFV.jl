
struct SymbolicIntervalGraph{F<:AbstractPolytope,N<:Number} <: LazySet{N}
    # TODO: what if input domain is not 1D ?
    Low
    Up
    domain::F
    lbs
    ubs
    var_los::Matrix{N}
    var_his::Matrix{N}
    var_ids
    max_vars::Int
    importance::Vector{N}
end


domain(s::SymbolicIntervalGraph) = s.domain
LazySets.radius(s::SymbolicIntervalGraph) = LazySets.radius(domain(s))
# last dim is batch-dim, which contains the input variables
LazySets.dim(s::SymbolicIntervalGraph) = size(s.Up)[1:end-1]

# just size of batch-dim, which contains input vars (and also the bias)
get_n_sym(s::SymbolicIntervalGraph) = size(s.Up)[end] - 1
get_n_in(s::SymbolicIntervalGraph) = dim(domain(s))

"""
Get the number of unfixed inputs in the domain of a symbolic interval.
"""
function get_n_unfixed(s::SymbolicIntervalGraph)
    input_set = domain(s)
    unfixed_mask = (low(input_set) .< high(input_set))
    return sum(unfixed_mask)
end

"""
Get the number of introduced fresh variables in a symbolic interval.
"""
function get_n_vars(s::SymbolicIntervalGraph)
    # there is one symbolic variable for each unfixed input and possibly
    # symbolic variables for fresh variables
    return get_n_sym(s) - get_n_unfixed(s)   
end

"""
Returns the symbolic interval described by the specified indices for the overapproximated values.

Note that indexing over the coefficients of the symbolic interval is not supported!
"""
function Base.getindex(s::SymbolicIntervalGraph, inds...)
    # TODO: what about max_vars (should there be less vars for the split interval?)
    # and var_ids (should we truncate zero entries?)
    # only use indices for first dimensions, last dimension is for coefficients
    Low = s.Low[inds...,:]
    Up  = s.Up[inds...,:]
    return init_symbolic_interval_graph(s, Low, Up)
end


"""
Size of a symbolic interval is (d₁,...,dₙ, N), where d₁,...,dₙ are the
dimensions of the quantity that is enclosed by the symbolic interval and
N is the number of the coefficients.
"""
function Base.size(s::SymbolicIntervalGraph)
    # return dim(s)
    return size(s.Up)
end


function Base.size(s::SymbolicIntervalGraph, d)
    #return dim(s)[d]
    return size(s.Up, d)
end


function Base.ndims(s::SymbolicIntervalGraph)
    #return length(dim(s))
    return length(size(s))
end


function Base.reshape(s::SymbolicIntervalGraph, sz; batch_included=true)
    sz = [sz...]
    if batch_included    
        # assume last dim is batch dimension sz = (d₁, ..., dₙ, N)
        @assert (sz[end] == 1) || (sz[end] == :) || (sz[end] == size(s)[end]) "Reshaping batch dimension is not supported!"
        # want linear coefficients to be in batch dimension
        sz[end] = size(s.Low, ndims(s.Low))
    else
        # sz is only data shapes sz = (d₁, ..., dₙ)
        # add batch dimension to shape
        sz = tuple(push!(sz, size(s.Low, ndims(s.Low))))
    end

    Low = reshape(s.Low, sz...)
    Up  = reshape(s.Up, sz...)

    return init_symbolic_interval_graph(s, Low, Up)
end


#= function Base.reshape(s::SymbolicIntervalGraph, dims)
    shape = [dims...]

    # if batch-dim already included in L.shape?
    if length(shape) == ndims(s.Low)
        # batch-dim included
        @assert shape[end] == 1 "Reshaping the batch-dimension is not allowed! (at node $(L.name))"
        shape[end] = size(s.Low, ndims(s.Low))
    else
        # batch-dim not included
        push!(shape, size(s.Low, ndims(s.Low)))
    end

    Low = reshape(s.Low, shape...)
    Up  = reshape(s.Up,  shape...)

    return init_symbolic_interval_graph(s, Low, Up)
end =#


function Base.reshape(s::SymbolicIntervalGraph, dims...)
    return reshape(s, dims)
end


function init_symbolic_interval_graph(net::CompGraph, input_set::AbstractHyperrectangle{N}; max_vars=10) where N<:Number
    n = dim(input_set)
    lbs = low(input_set)
    ubs = high(input_set)
    # only want variable for those inputs, that are not fixed (i.e. ub is strictly larger than lb)
    fixed = (lbs .== ubs)
    Low = [I(n)[:,.~fixed] lbs .* fixed]
    Up  = [I(n)[:,.~fixed] ubs .* fixed]

    # dict "layername" => [lbs]
    lbs = Dict()
    ubs = Dict()

    n_unfixed = sum(.~fixed)
    var_los = zeros(N, max_vars, n_unfixed + 1)
    var_his = zeros(N, max_vars, n_unfixed + 1)

    importance = zeros(N, dim(input_set))

    # maybe use [("layername", position)]
    var_ids = []

    return SymbolicIntervalGraph(Low, Up, input_set, lbs, ubs, var_los, var_his, 
                                 var_ids, max_vars, importance)
end


"""
Initialises a symbolic interval to a given shape given Hyperrectangular input set.
"""
function init_symbolic_interval_graph(net::CompGraph, input_set::AbstractHyperrectangle{N}, input_shape::Tuple{Vararg{<:Integer}}; max_vars=10) where N <: Number
    ŝ = init_symbolic_interval_graph(net, input_set, max_vars=max_vars)
    return reshape(ŝ, input_shape)
end


"""
Creates a new symbolic interval from an old one with a new input set.
"""
function init_symbolic_interval_graph(s::SymbolicIntervalGraph, input_set::AbstractHyperrectangle{N}; max_vars=10) where N <: Number
    n = dim(input_set)
    lbs = low(input_set)
    ubs = high(input_set)
    # only want variable for those inputs, that are not fixed (i.e. ub is strictly larger than lb)
    fixed = (lbs .== ubs)
    Low = [I(n)[:,.~fixed] lbs .* fixed]
    Up  = [I(n)[:,.~fixed] ubs .* fixed]

    # TODO: is copy enough?
    # dict "layername" => [lbs]
    lbs = copy(s.lbs)
    ubs = copy(s.ubs)

    n_unfixed = sum(.~fixed)
    var_los = zeros(N, max_vars, n_unfixed + 1)
    var_his = zeros(N, max_vars, n_unfixed + 1)

    importance = zeros(N, dim(input_set))

    # maybe use [("layername", position)]
    var_ids = []

    return SymbolicIntervalGraph(Low, Up, input_set, lbs, ubs, var_los, var_his, 
                                 var_ids, max_vars, importance) 
end


"""
Takes an existing SymbolicIntervalGraph and generates a new instance thats different 
only in its symbolic lower and upper bound.

args:
    s - the existing symbolic interval
    Low - (tensor) the new symbolic lower bound
    Up - (tensor) the new symbolic upper bound
"""
function init_symbolic_interval_graph(s::SymbolicIntervalGraph, Low, Up)
    return SymbolicIntervalGraph(Low, Up, s.domain, s.lbs, s.ubs, s.var_los, s.var_his,
                                 s.var_ids, s.max_vars, s.importance)
end


"""
Computes symbolic lower and upper bounds without occurrences of fresh variables
by substituting them with their bounds depending on the input variables.

args:
    s - symbolic interval with fresh variables to be substituted

returns:
    subs_lb - symbolic lower bound without fresh variables
    subs_ub - symbolic upper bound without fresh variables
"""
function substitute_variables(s::SymbolicIntervalGraph)
    n_in = get_n_in(s)
    n_unfixed = get_n_unfixed(s)
    n_vars = get_n_vars(s)

    shape = size(s.Low)
    Low = reshape(s.Low, (prod(shape[1:end-1]), shape[end]))
    Up  = reshape(s.Up, (prod(shape[1:end-1]), shape[end]))

    subs_lb, subs_ub = substitute_variables(Low, Up, s.var_los, s.var_his, n_unfixed, n_vars)

    subs_lb = reshape(subs_lb, shape[1:end-1]..., :)
    subs_ub = reshape(subs_ub, shape[1:end-1]..., :)

    return subs_lb, subs_ub
end


"""
Returns point x* in the input domain that maximizes the upper bound of the symbolic interval.

s.t. s.Up[:,1:end-1] * x* + s.Up[:,end] = upper_bounds(s)
!!! the equation only holds if all input variables are unfixed !!!
otherwise fixed dimensions need to be removed from x* 
"""
function maximizer(s::SymbolicIntervalGraph)
    subs_sym_lo, subs_sym_hi = substitute_variables(s)

    lbs = low(domain(s))
    ubs = high(domain(s))
    unfixed_mask = (lbs .< ubs)

    x̂_star = maximizer(subs_sym_lo, lbs[unfixed_mask], ubs[unfixed_mask])
    # at fixed indices, lb == ub, so it doesn't matter
    x_star = lbs
    if ndims(x̂_star) > 1
        x_star = repeat(lbs', size(x̂_star, 1), 1)
        x_star[:, unfixed_mask] .= x̂_star
    else
        x_star[unfixed_mask] .= x̂_star
    end
    
    return x_star
end


"""
Returns point x* in the input domain that minimizes the lower bound of the symbolic interval.

s.t. s.Low[:,1:end-1] * x* + s.Low[:,end] = lower_bounds(s)
!!! the equation only holds if all input variables are unfixed !!!
otherwise fixed dimensions need to be removed from x* 

For multiple dimensions, x*[i,:] is the minimizer of the i-th dimension
"""
function minimizer(s::SymbolicIntervalGraph)
    subs_sym_lo, subs_sym_hi = substitute_variables(s)

    lbs = low(domain(s))
    ubs = high(domain(s))
    unfixed_mask = (lbs .< ubs)

    x̂_star = minimizer(subs_sym_lo, lbs[unfixed_mask], ubs[unfixed_mask])
    # at fixed indices, lb == ub, so it doesn't matter
    x_star = lbs
    if ndims(x̂_star) > 1
        x_star = repeat(lbs', size(x̂_star, 1), 1)
        x_star[:, unfixed_mask] .= x̂_star
    else
        x_star[unfixed_mask] .= x̂_star
    end
    return x_star
end


"""
Calculates concrete lower and upper bounds given a symbolic interval in the 
shape of the symbolic interval.
"""
function bounds(s::SymbolicIntervalGraph)
    Low = Flux.flatten(s.Low)
    Up  = Flux.flatten(s.Up)
    n_neurons = size(Low, 1)
    n_in = get_n_in(s)
    n_unfixed = get_n_unfixed(s)
    current_n_vars = get_n_vars(s)

    # need n_unfixed instead of n_in, because only unfixed inputs are modeled as variables
    subs_LL, subs_UU = substitute_variables(Low, Up, s.var_los, s.var_his, n_unfixed, current_n_vars)


    l_bounds = bounds(subs_LL, domain(s))
    u_bounds = bounds(subs_UU,  domain(s))

    # take first of [ll, lu] and second of [ul, uu] for outer bounds
    return reshape(l_bounds[1], size(s)[1:end-1]), reshape(u_bounds[2], size(s)[1:end-1])
end


"""
Adds a constant to the equations' biases.

Constant offset is stored in the equations batch dimension at the last entry.

!!! inplace method !!!
"""
function add_constant!(eqs::AbstractArray, c)
    nd = ndims(eqs)
    batch_size = size(eqs, nd)
    selectdim(eqs, nd, batch_size) .+= c
end

"""
Adds a constant to a symbolic interval.

!!! inplace method !!!
"""
function add_constant!(s::SymbolicIntervalGraph, c)
    nd = ndims(s.Low)
    batch_size = size(s.Low, nd)
    # constant offset is the last entry in the batch dimension
    # (batch dimension holds the coeffs for the variables)
    selectdim(s.Low, nd, batch_size) .+= c
    selectdim(s.Up,  nd, batch_size) .+= c
end

function add_constant(s::SymbolicIntervalGraph, c)
    Low = copy(s.Low)
    Up  = copy(s.Up)

    nd = ndims(s.Low)
    batch_size = size(s.Low, nd)
    # constant offset is the last entry in the batch dimension
    # (batch dimension holds the coeffs for the variables)
    selectdim(Low, nd, batch_size) .+= c
    selectdim(Up,  nd, batch_size) .+= c

    return init_symbolic_interval_graph(s, Low, Up)
end



"""
Bisect input interval of dimension i of domain of symbolic interval.

returns:
    [s₁, s₂] - list of two symbolic intervals with both halves of the domain.
"""
function split_symbolic_interval_graph(s::SymbolicIntervalGraph{<:Hyperrectangle}, index::Int)
    domain1, domain2 = split(domain(s), index)

    current_n_vars = get_n_vars(s)
    # can't have more vars than parent node?
    s1 = init_symbolic_interval_graph(s, domain1; max_vars=current_n_vars)
    s2 = init_symbolic_interval_graph(s, domain2; max_vars=current_n_vars)

    return [s1, s2]
end


"""
Bisect input interval of dimension with largest radius in domain of symbolic interval.
"""
function split_largest_interval(s::SymbolicIntervalGraph)
    largest_dimension = argmax(high(domain(s)) - low(domain(s)))
    return split_symbolic_interval_graph(s, largest_dimension)
end


"""
Splitting based on intermediate importance scores (i.e. coefficients in all crossing ReLUs)
"""
function split_important_interval(s::SymbolicIntervalGraph{<:Hyperrectangle})
    radius = high(domain(s)) - low(domain(s))
    # if there are no more crossing ReLUs, importance will be zero vector -> if we
    # multiply it with radius, all inputs are equally important, and argmax always
    # returns the first index -> infinitely loop splitting
    most_important_dim = sum(s.importance) == 0. ? argmax(radius) : argmax(s.importance .* radius)
    return split_symbolic_interval_graph(s, most_important_dim)
end


"""
Split input domain of symbolic interval n times using a specified split technique.

args:
    cell - symbolic interval to split
    n - number of splits

kwargs:
    split - splitting method (default: split_largest_interval)

returns:
    queue containing symbolic intervals with split domain
"""
function NPO.split_multiple_times(cell::SymbolicIntervalGraph, n; split=split_largest_interval)
    q = Queue{SymbolicIntervalGraph}()
    enqueue!(q, cell)
    for i in 1:n
        new_cells = split(dequeue!(q))
        enqueue!(q, new_cells[1])
        enqueue!(q, new_cells[2])
    end
    return q
end

