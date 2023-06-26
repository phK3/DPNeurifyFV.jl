
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
# FIXME: not working anymore, what if there are lots of fixed input vars?
get_n_vars(s::SymbolicIntervalGraph) = get_n_sym(s) - get_n_in(s)


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


function Base.size(s::SymbolicIntervalGraph)
    return dim(s)
end


function Base.size(s::SymbolicIntervalGraph, d)
    return dim(s)[d]
end


function Base.ndims(s::SymbolicIntervalGraph)
    return length(dim(s))
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

    var_los = zeros(N, max_vars, dim(input_set) + 1)
    var_his = zeros(N, max_vars, dim(input_set) + 1)

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
    n_vars = get_n_vars(s)

    shape = size(s.Low)
    Low = reshape(s.Low, (prod(shape[1:end-1]), shape[end]))
    Up  = reshape(s.Up, (prod(shape[1:end-1]), shape[end]))

    subs_lb, subs_ub = substitute_variables(Low, Up, s.var_los, s.var_his, n_in, n_vars)

    subs_lb = reshape(subs_lb, shape)
    subs_ub = reshape(subs_ub, shape)

    return subs_lb, subs_ub
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

