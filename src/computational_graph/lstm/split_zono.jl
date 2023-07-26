

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
    splits::Dict{String, <:AbstractVector{<:Tuple{<:Integer, <:Real, <:Real}}}
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
    splits = Dict{String, Vector{Tuple{Int64, Float64, Float64}}}()
    generator_map = [("input", i) for (i, _) in enumerate(generators(input_set))]
    split_A = Array{N}(undef, 0, 0)
    split_b = Array{N}(undef, 0)
    return SplitZonotope(z, splits, generator_map, split_A, split_b, shape)
end  


function SplitZonotope(input_set::AbstractHyperrectangle{N}) where N <: Number
    shape = size(low(input_set))
    return SplitZonotope(input_set, shape)
end


function SplitZonotope(ẑ::Zonotope{N}, sz::SplitZonotope{N}) where N <: Number
    return SplitZonotope(ẑ, sz.splits, sz.generator_map, sz.split_A, sz.split_b, sz.shape)
end


function SplitZonotope(ẑ::Zonotope{N}, sz::SplitZonotope{N}, shape) where N <: Number
    return SplitZonotope(ẑ, sz.splits, sz.generator_map, sz.split_A, sz.split_b, shape)
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
    return SplitZonotope(ẑ, sz.splits, sz.generator_map, split_A, split_b, sz.shape)
end


function Base.getindex(sz::SplitZonotope{N}, inds::AbstractArray{<:Integer}) where N <: Number
    z = sz.z
    ẑ = Zonotope(z.center[inds], z.generators[inds, :])
    return SplitZonotope(ẑ, sz.splits, sz.generator_map, sz.split_A, sz.split_b)
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
    return SplitZonotope(ẑ, sz.splits, sz.generator_map, sz.split_A, sz.split_b, shape)
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

    return SplitZonotope(z, merge(sz₁.splits, sz₂.splits), generator_map, split_A, split_b, sz₁.shape)
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


function hadamard_prod(a::AbstractVector{N}, sz::SplitZonotope{N}) where N <: Number
    ẑ = Zonotope(a .* sz.z.center, a .* sz.z.generators)
    return SplitZonotope(ẑ, sz.splits, sz.generator_map, sz.split_A, sz.split_b, sz.shape)
end