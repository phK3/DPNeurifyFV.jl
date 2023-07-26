

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
    H = overapproximate(z, Hyperrectangle)
    l = low(H)
    u = high(H)

    # adding split constraints
    if L.name in keys(sz.splits)
        split_idxs, l_split, u_split = unzip(sz.splits[L.name])
        split_idxs = _vec(split_idxs)
        l_split = _vec(l_split)
        u_split = _vec(u_split)

        # we need those splits, others already hold by bounds from input zonotope
        lmask = l[split_idxs] .< l_split
        umask = u[split_idxs] .> u_split

        # x ≥ l ↔ -x ≤ -l 
        A_l = .- z.generators[split_idxs[lmask], :]
        b_l = .- l_split[lmask]

        A_u = z.generators[split_idxs[umask], :]
        b_u = u_split[umask]

        A = [A_l; A_u]
        b = [b_l; b_u]

        # TODO: right number of generators?
        Â, b̂ = get_constraints_matrix(sz, A, b)


        # now can adjust bounds for propagation
        l[split_idxs] .= max.(l[split_idxs], l_split)
        u[split_idxs] .= min.(u[split_idxs], u_split)
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

    sẑ = SplitZonotope(ẑ, sz.splits, sz.generator_map, Â, b̂, sz.shape)
    push!(sẑ.generator_map, [(L.name, i) for i in (1:dim(z))[crossing]]...)
    return sẑ
end


function forward_node(solver::LSTMSolver, L::Flatten, sz::SplitZonotope{N}) where N <: Number
    c = reshape(sz.z.center, sz.shape)
    ĉ = reshape(c, :, size(c)[end])
    return SplitZonotope(sz.z, sz.splits, sz.generator_map, sz.split_A, sz.split_b, size(ĉ))
end