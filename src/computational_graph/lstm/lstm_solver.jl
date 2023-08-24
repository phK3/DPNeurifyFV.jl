

struct LSTMSolver <: Solver end


function forward_node(solver::LSTMSolver, L::Linear, sz::SplitZonotope)
    # TODO: use L.dense, but somehow ignore the bias?
    # TODO: figure out way for Float32/Float64 conversion
    ẑ = LazySets.affine_map(Float64.(L.dense.weight), sz.z, Float64.(L.dense.bias))

    ĉ = L.dense(reshape(sz.z.center, sz.shape))
    shape = size(ĉ)

    return SplitZonotope(ẑ, sz, shape)
end


function forward_node(solver::LSTMSolver, L::AddConst, sz::SplitZonotope)
    c = reshape(sz.z.center, sz.shape)
    # only store zonotopes for vectors
    ĉ = vec(c .+ L.c)
    return SplitZonotope(Zonotope(ĉ, sz.z.generators), sz)
end


function forward_node(solver::LSTMSolver, L::Relu, sz::SplitZonotope)
    z = sz.z
    l, u = get_bounds!(sz, L.name)

    # adding split constraints
    if L.name in keys(sz.splits)
        l, u, Â, b̂ = get_split_bounds(sz, sz.splits[L.name], l, u)
    else
        Â = sz.split_A
        b̂ = sz.split_b
    end

    crossing = l .< 0 .&& u .> 0
    λ = relaxed_relu_gradient.(l, u)
    β = 0.5 .* (.- λ .* l)
    
    ĉ = λ .* z.center .+ max.(0, β)
    Ĝ = λ .* z.generators
    E = β .* I(dim(z))[:, crossing]
    ẑ = Zonotope(ĉ, [Ĝ E])

    # register new generators in generator_map

    sẑ = SplitZonotope(ẑ, sz.splits, sz.bounds, sz.generator_map, Â, b̂, sz.shape)
    push!(sẑ.generator_map, [(L.name, i) for i in (1:dim(z))[crossing]]...)
    return sẑ
end


function forward_node(solver::LSTMSolver, L::Flatten, sz::SplitZonotope{N}) where N <: Number
    c = reshape(sz.z.center, sz.shape)
    ĉ = reshape(c, :, size(c)[end])
    return SplitZonotope(sz.z, sz.splits, sz.bounds, sz.generator_map, sz.split_A, sz.split_b, size(ĉ))
end