

"""
Zonotope that carries splitting information on the nodes of the network.

The underlying zonotope is always stored as representing a vector x ∈ ℝⁿ (needs to be reshaped for tensor ops)

splits are stored as a dictionary, where splits (neuron_idx, split_value) are stored for each layer identified by its name

We also store for each generator (each column of the generator matrix) at which neuron it was introduced.
"""
struct SplitZonotope{N} <: AbstractZonotope{N}
    z::Zonotope{N}
    # mapping layer_name -> [(neuron_idx, lb, ub)]
    # for each split neuron in a layer, we store its lower bound and upper bound (that were set by the split)
    # splits::Dict{String, <:AbstractVector{<:Tuple{<:Integer, <:Real, <:Real}}}
    # TODO: find right type for allowing (idx, l, u) and (idx1, idx2, l, u)
    splits::Dict{String, <:AbstractVector}
    # mapping layer_name -> [(lb, ub)]
    # for each layer, we store the best lower and upper bounds, we've seen so far
    # TODO: find type for allowing (l, u) and (lx, ux, ly, uy)
    # bounds::Dict{String, <:Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}}
    bounds::Dict{String, <:Tuple{Vararg{Vector{N}}}}
    # mapping column idx in zonotope -> (layer_name, neuron_idx)
    generator_map::Vector{<:Tuple{String, <:Integer}}
    # constraints added to LP solver later on (for now just in form Ax ≤ b)
    split_A::AbstractMatrix{N}
    split_b::AbstractVector{N}
    shape
end


function SplitZonotope(input_set::AbstractHyperrectangle{N}, shape) where N <: Number
    z = convert(Zonotope, input_set)
    # z = zono2float32(z)
    # splits = Dict{String, Vector{Tuple{Int64, Float64, Float64}}}()
    splits = Dict{String, Vector}()
    # bounds = Dict{String, Vector{Tuple{Float64, Float64}}}()
    # bounds = Dict{String, Tuple{Vector{Float64}}}()
    bounds = Dict{String, Tuple{Vararg{Vector{N}}}}()
    bounds["input"] = (low(input_set), high(input_set))
    generator_map = [("input", i) for (i, _) in enumerate(generators(input_set))]
    split_A = Array{N}(undef, 0, 0)
    split_b = Array{N}(undef, 0)
    return SplitZonotope(z, splits, bounds, generator_map, split_A, split_b, shape)
end  


function SplitZonotope(input_set::AbstractHyperrectangle{N}) where N <: Number
    shape = size(low(input_set))
    return SplitZonotope(input_set, shape)
end


function SplitZonotope(ẑ::Zonotope{N}, sz::SplitZonotope{N}) where N <: Number
    return SplitZonotope(ẑ, sz.splits, sz.bounds, sz.generator_map, sz.split_A, sz.split_b, sz.shape)
end


function SplitZonotope(ẑ::Zonotope{N}, sz::SplitZonotope{N}, shape) where N <: Number
    return SplitZonotope(ẑ, sz.splits, sz.bounds, sz.generator_map, sz.split_A, sz.split_b, shape)
end


LazySets.ngens(sz::SplitZonotope) = ngens(sz.z)


"""
Returns a SplitZonotope representing the 0ⁿ vector with a zero generator matrix
with the same number of generators as the input SplitZonotope.
"""
function Base.zero(sz::SplitZonotope{N}) where N <: Number
    z = sz.z
    c = z.center
    m, n = size(z.generators)
    # need to reshape, so G is a matrix
    ẑ = Zonotope(zero(c), zeros(m, n))

    # no split constraints in zero
    split_A = Array{N}(undef, 0, 0)
    split_b = Array{N}(undef, 0)
    return SplitZonotope(ẑ, sz.splits, sz.bounds, sz.generator_map, split_A, split_b, sz.shape)
end


function Base.getindex(sz::SplitZonotope{N}, inds::AbstractArray{<:Integer}) where N <: Number
    z = sz.z
    ẑ = Zonotope(z.center[inds], z.generators[inds, :])
    return SplitZonotope(ẑ, sz.splits, sz.bounds, sz.generator_map, sz.split_A, sz.split_b)
end


"""
Gets SplitZonotope corresponding to a more than one-dimensional index.

If SplitZonotope encloses quantity x with shape (d₁, d₂), then 
get_tensor_idx(sz, [d₁, d₂], :, 1) will return the part of the SplitZonotope that
encloses x[:,1].
"""
function get_tensor_idx(sz::SplitZonotope{N}, dims, inds...) where N <: Number
    # need to have list comprehension construct as : is allowed in indexing
    #free_dims = length(dims) - length([e for e in inds if typeof(e) <: Integer])
    free_dims = length(dims) - length([e for e in inds if e != :])
    # should we also support scalar output (i.e. 0 free dims)?
    @assert free_dims == 1 "Can only generate zonotope for vector ouput, but got $free_dims free dimensions!"
    
    z = sz.z

    c = z.center
    G = z.generators
    
    n_gens = size(G, 2)
    ĉ = reshape(c, dims...)
    Ĝ = reshape(G, dims..., n_gens)

    shape = size(ĉ[inds...])
    
    # Ĝ[inds..., :] since we want all generators
    ẑ = Zonotope(ĉ[inds...], Ĝ[inds..., :])
    return SplitZonotope(ẑ, sz.splits, sz.bounds, sz.generator_map, sz.split_A, sz.split_b, shape)
end


"""
Addition of two zonotopes that have some generators in common.

The output zonotope will have generators G = [G_common G₁ G₂]
where Gᵢ are the generators that occur only in zonotope i and their order is not changed.

args:
    common_gens₁ - idxs of shared generators in z₁
    common_gens₂ - idxs of shared generators in z₂
"""
function direct_sum(z₁::Zonotope{N}, z₂::Zonotope{N}, common_gens₁, common_gens₂) where N <: Number
    c₁ = z₁.center
    G₁ = z₁.generators
    c₂ = z₂.center
    G₂ = z₂.generators
    
    @assert length(c₁) == length(c₂) "Zonotopes must have equal dimension! Got dim(z₁) = $(length(c₁)), dim(z₂) = $(length(c₂))"
    @assert length(common_gens₁) == length(common_gens₂) "Number of common generators has to match!"
    
    distinct₁ = setdiff(1:size(G₁, 2), common_gens₁)
    distinct₂ = setdiff(1:size(G₂, 2), common_gens₂)
    
    G_common = G₁[:,common_gens₁] .+ G₂[:,common_gens₂]
    return Zonotope(c₁ .+ c₂, [G_common G₁[:,distinct₁] G₂[:,distinct₂]])   
end


function direct_sum(sz₁::SplitZonotope{N}, sz₂::SplitZonotope{N}) where N <: Number
    common_gens₁ = filter(!isnothing, indexin(sz₂.generator_map, sz₁.generator_map))
    common_gens₂ = filter(!isnothing, indexin(sz₁.generator_map, sz₂.generator_map))

    z = direct_sum(sz₁.z, sz₂.z, common_gens₁, common_gens₂)

    distinct₁ = setdiff(1:ngens(sz₁), common_gens₁)
    distinct₂ = setdiff(1:ngens(sz₂), common_gens₂)
    generator_map = [sz₁.generator_map[common_gens₁]; sz₁.generator_map[distinct₁]; sz₂.generator_map[distinct₂]]

    if size(sz₁.split_A, 1) == 0
        # there are no split constraints in sz₁
        split_A = sz₂.split_A
        split_b = sz₂.split_b
    elseif size(sz₂.split_A, 1) == 0
        # there are no split constraints in sz₂
        split_A = sz₁.split_A
        split_b = sz₁.split_b
    else
        split_A = [sz₁.split_A[:, common_gens₁] sz₁.split_A[:, distinct₁] zeros(size(sz₁.split_A, 1), length(distinct₂));
                sz₂.split_A[:, common_gens₂] zeros(size(sz₂.split_A, 1), length(distinct₂)) sz₂.split_A[:, distinct₁]]
        split_b = [sz₁.split_b; sz₂.split_b]
    end

    # TODO: do both split zonotopes really necessarily have the same bounds?
    return SplitZonotope(z, merge(sz₁.splits, sz₂.splits), sz₁.bounds, generator_map, split_A, split_b, sz₁.shape)
end


"""
Expand generators, s.t. sz₁ and sz₂ have the same set of generators.

returns:
    ŝz₁ - SplitZonotope representing the same set as sz₁, but with 
            added empty generators for distinct generators of sz₂
    ŝz₂ - SplitZonotope representing the same set as sz₂, but with 
            added empty generators for distinct generators of sz₁
"""
function expand_generators(sz₁::SplitZonotope, sz₂::SplitZonotope)
    common_gens₁ = filter(!isnothing, indexin(sz₂.generator_map, sz₁.generator_map))
    common_gens₂ = filter(!isnothing, indexin(sz₁.generator_map, sz₂.generator_map))
    distinct₁ = setdiff(1:ngens(sz₁), common_gens₁)
    distinct₂ = setdiff(1:ngens(sz₂), common_gens₂)
    generator_map = [sz₁.generator_map[common_gens₁]; sz₁.generator_map[distinct₁]; sz₂.generator_map[distinct₂]]

    m₁ = size(sz₁.z.center, 1)
    m₂ = size(sz₂.z.center, 1)
    z₁ = Zonotope(sz₁.z.center, [sz₁.z.generators[:, common_gens₁] sz₁.z.generators[:, distinct₁] zeros(m₁, length(distinct₂))])
    z₂ = Zonotope(sz₂.z.center, [sz₂.z.generators[:, common_gens₂] zeros(m₂, length(distinct₁)) sz₂.z.generators[:, distinct₂]])

    ŝz₁ = SplitZonotope(z₁, sz₁.splits, sz₁.bounds, generator_map, sz₁.split_A, sz₁.split_b, sz₁.shape)
    ŝz₂ = SplitZonotope(z₂, sz₂.splits, sz₂.bounds, copy(generator_map), sz₂.split_A, sz₂.split_b, sz₂.shape)

    return ŝz₁, ŝz₂
end



function hadamard_prod(a::AbstractVector{N}, sz::SplitZonotope{N}) where N <: Number
    ẑ = Zonotope(a .* sz.z.center, a .* sz.z.generators)
    return SplitZonotope(ẑ, sz.splits, sz.bounds, sz.generator_map, sz.split_A, sz.split_b, sz.shape)
end


"""
Returns constraints Â x ≤ b̂ when adding constraint A x ≤ b to already existing constraints in the SplitZonotope
"""
function get_constraints_matrix(sz::SplitZonotope, A::AbstractMatrix, b::AbstractVector)
    m, n = size(sz.split_A)
    m̂, n̂ = size(A)
    if n < n̂
        # old constraints don't have all generators
        Â = [sz.split_A zeros(m, n̂ - n);
             A]
    elseif n > n̂
        # new constraints don't have all generators
        # TODO: when could this happen?
        Â = [sz.split_A;
             sz.A zeros(m̂, n - n̂)]
    else
        Â = [sz.split_A; A]
    end

    b̂ = [sz.split_b; b]

    return Â, b̂   
end


"""
Compares the current bounds to the best know bounds up to now and 
returns the tightest of both elementwise.
Also updates the tightest bounds stored in the bounds dictionary.

WARNING: If you call get_bounds!(sz, "other_layer") where sz doesn't
represent an overapproxmation of other layer, then you will set the
bouns of other layer to the current bounds!!!

args:
    sz - SplitZonotope we want the bounds for
    layer_name - name of the layer, we want the bounds for 
            (needed for retrieving and updating the best known bounds)

returns:
    l - vector of lower bounds
    u - vector of upper bounds
"""
function get_bounds!(sz::SplitZonotope, layer_name::String)
    H = overapproximate(sz.z, Hyperrectangle)
    l = low(H)
    u = high(H)

    if layer_name in keys(sz.bounds)
        l̂, û = sz.bounds[layer_name]
        l = max.(l, l̂)
        u = min.(u, û)
    end

    # copy s.t. modifying the returned l, u doesn't modify the stored bounds
    sz.bounds[layer_name] = (copy(l), copy(u))
    return l, u        
end


function get_bounds!(sx::SplitZonotope, sy::SplitZonotope, layer_name::String)
    Hx = overapproximate(sx.z, Hyperrectangle)
    Hy = overapproximate(sy.z, Hyperrectangle)
    lx = low(Hx)
    ux = high(Hx)
    ly = low(Hy)
    uy = high(Hy)

    # both sx and sy should have the same stored bounds
    if layer_name in keys(sx.bounds)
        # for 2d functions, store (lx, ux, ly, uy)
        l̂x, ûx, l̂y, ûy = sx.bounds[layer_name]
        lx = max.(lx, l̂x)
        ux = min.(ux, ûx)
        ly = max.(ly, l̂y)
        uy = min.(uy, ûy)
    end

    # TODO: is it the same object/reference as sy.bounds? (we want it to be that way)
    sx.bounds[layer_name] = (copy(lx), copy(ux), copy(ly), copy(uy))
    return lx, ux, ly, uy        
end



"""
Calculates linear constraints Ax ≤ b induced by the splits on the current layer
as well as the lower and upper bounds on the neurons (that might be improved due to 
the splits).

args:
    sz - SplitZonotope for the current layer
    splits - neuron splits for the current layer
    l - vector of concrete lower bounds
    u - vector of concrete upper bounds

returns:
    l - vector of lower bounds for the neurons in the current layer (maybe tightened from split)
    u - vector of upper bounds for the neurons in the current layer (maybe tightened from splits)
    A - constraint matrix for enforcing splits in LP later on
    b - constraint vector for enforcing splits in LP later on
"""
function get_split_bounds(sz::SplitZonotope, splits::AbstractVector{<:Tuple{<:Integer, <:Real, <:Real}}, l::AbstractVector, u::AbstractVector)   
    z = sz.z

    split_idxs, l_split, u_split = unzip(splits)
    split_idxs = _vec(split_idxs)
    l_split = _vec(l_split)
    u_split = _vec(u_split)

    # if l_split (the lower bound enforced by the split) is larger than the 
    # lower bound from the zonotope, we need that split
    lmask = l[split_idxs] .< l_split
    umask = u[split_idxs] .> u_split

    # x ≥ l ↔ -x ≤ -l
    A_l = .- z.generators[split_idxs[lmask], :]
    b_l = .- l_split[lmask]
    A_u = z.generators[split_idxs[umask], :]
    b_u = u_split[umask]

    A = [A_l; A_u]
    b = [b_l; b_u]
    Â, b̂ = get_constraints_matrix(sz, A, b)

    # take better of bound calculated from zono and bound stored in split
    l[split_idxs] .= max.(l[split_idxs], l_split)
    u[split_idxs] .= min.(u[split_idxs], u_split)

    return l, u, Â, b̂
end


function get_split_bounds(sz::SplitZonotope, splits::AbstractVector{<:Tuple{<:Integer, <:Real, <:Real}})
    z = sz.z

    H = overapproximate(z, Hyperrectangle)
    l = low(H)
    u = high(H)
    return get_split_bounds(sz, splits, l, u)
end


"""
Calculates linear constraints Ax ≤ b induced by the splits on the current layer
as well as the lower and upper bounds on the neurons (that might be improved due to 
the splits).
Also updates the tightest known lower and upper bounds up until now.

args:
    sz - SplitZonotope for the current layer
    splits - neuron splits for the current layer
    layer_name - name of the current layer

returns:
    l - vector of lower bounds for the neurons in the current layer (maybe tightened from split)
    u - vector of upper bounds for the neurons in the current layer (maybe tightened from splits)
    A - constraint matrix for enforcing splits in LP later on
    b - constraint vector for enforcing splits in LP later on
"""
function get_split_bounds!(sz::SplitZonotope, splits::AbstractVector{<:Tuple{<:Integer, <:Real, <:Real}}, layer_name::String)
    l, u = get_bounds!(sz, layer_name)
    return get_split_bounds(sz, splits, l, u)
end


"""
Calculates lower and upper bounds as well as linear constraints for functions of two inputs.
Sets for both inputs need to be represented as SplitZonotopes.

args:
    sx - SplitZonotope overapproximating first input
    sy - SplitZonotope overapproximating second input
    splits - vector of tuples (i, lx, ux, ly, uy) indicating bounds on the input of neuron i for x and y

returns:
    lx - lower bound on first input (possibly tightened by split)
    ux - upper bound on first input (possibly tightened by split)
    ly - lower bound on second input (possibly tightened by split)
    uy - upper bound on second input (possibly tightened by split)
    A - constraint matrix for enforcing splits in LP later on
    b - constraint vector for enforcing splits in LP later on
"""
function get_split_bounds(sx::SplitZonotope, sy::SplitZonotope, splits::AbstractVector)
    x_splits = map(x -> (x[1], x[2], x[3]), splits)
    y_splits = map(x -> (x[1], x[4], x[5]), splits)

    lx, ux, A₁, b₁ = get_split_bounds(sx, x_splits)
    ly, uy, A₂, b₂ = get_split_bounds(sy, y_splits)

    A = [A₁; A₂]
    b = [b₁; b₂]

    return lx, ux, ly, uy, A, b
end


function get_split_bounds!(sx::SplitZonotope, sy::SplitZonotope, splits::AbstractVector, layer_name::String)
    x_splits = map(x -> (x[1], x[2], x[3]), splits)
    y_splits = map(x -> (x[1], x[4], x[5]), splits)

    lx, ux, ly, uy = get_bounds!(sx, sy, layer_name)
    lx, ux, A₁, b₁ = get_split_bounds!(sx, x_splits, lx, ux)
    ly, uy, A₂, b₂ = get_split_bounds!(sy, y_splits, ly, uy)

    A = [A₁; A₂]
    b = [b₁; b₂]

    return lx, ux, ly, uy, A, b   
end
