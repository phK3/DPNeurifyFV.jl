

"""
Zonotope that carries splitting information on the nodes of the network.

The underlying zonotope is always stored as representing a vector x ∈ ℝⁿ (needs to be reshaped for tensor ops)

splits are stored as a dictionary, where splits (neuron_idx, split_value) are stored for each layer identified by its name

We also store for each generator (each column of the generator matrix) at which neuron it was introduced.
"""
struct SplitZonotope{N} <: AbstractZonotope{N}
    z::Zonotope{N}
    # mapping layer_name -> (neuron_idx, split_value)
    splits
    # mapping column idx in zonotope -> (layer_name, neuron_idx)
    generator_map
end


function SplitZonotope(input_set::AbstractHyperrectangle{N}) where N <: Number
    z = convert(Zonotope, input_set)
    splits = Dict()
    generator_map = [("input", i) for (i, _) in enumerate(generators(input_set))]
    return SplitZonotope(z, splits, generator_map)
end


LazySets.ngens(sz::SplitZonotope) = ngens(sz.z)


function Base.zero(sz::SplitZonotope{N}) where N <: Number
    z = sz.z
    c = z.center
    n = length(c)
    # need to reshape, so G is a matrix
    ẑ = Zonotope(zero(c), reshape(zero(c), n, 1))
    return SplitZonotope(ẑ, sz.splits, sz.generator_map)
end


function Base.getindex(sz::SplitZonotope{N}, inds::AbstractArray{<:Integer}) where N <: Number
    z = sz.z
    ẑ = Zonotope(z.center[inds], z.generators[inds, :])
    return SplitZonotope(ẑ, sz.splits, sz.generator_map)
end


"""
Gets SplitZonotope corresponding to a more than one-dimensional index.

If SplitZonotope encloses quantity x with shape (d₁, d₂), then 
get_tensor_idx(sz, [d₁, d₂], :, 1) will return the part of the SplitZonotope that
encloses x[:,1].
"""
function get_tensor_idx(sz::SplitZonotope{N}, dims, inds...) where N <: Number
    # need to have list comprehension construct as : is allowed in indexing
    free_dims = length(dims) - length([e for e in inds if typeof(e) <: Integer])
    # should we also support scalar output (i.e. 0 free dims)?
    @assert free_dims == 1 "Can only generate zonotope for vector ouput, but got $free_dims free dimensions!"
    
    z = sz.z

    c = z.center
    G = z.generators
    
    n_gens = size(G, 2)
    ĉ = reshape(c, dims...)
    Ĝ = reshape(G, dims..., n_gens)
    
    # Ĝ[inds..., :] since we want all generators
    ẑ = Zonotope(ĉ[inds...], Ĝ[inds..., :])
    return SplitZonotope(ẑ, sz.splits, sz.generator_map)
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

    return SplitZonotope(z, merge(sz₁.splits, sz₂.splits), generator_map)
end


function hadamard_prod(a::AbstractVector{N}, sz::SplitZonotope{N}) where N <: Number
    ẑ = Zonotope(a .* sz.z.center, a .* sz.z.generators)
    return SplitZonotope(ẑ, sz.splits, sz.generator_map)
end