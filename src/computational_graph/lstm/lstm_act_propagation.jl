

function propagate_σ_y(sx::SplitZonotope, sy::SplitZonotope, name::String; n_samples=100, use_zono=false, method=:remezlike)
    x = sx.z
    y = sy.z

    lx, ux, ly, uy = get_bounds!(sx, sy, name)

    # splits should be the same in both!
    if name in keys(sx.splits)
        #idx1, idx2, l_split, u_split = unzip(sx.splits[name])
        lx, ux, ly, uy, Â, b̂ = get_split_bounds(sx, sy, sx.splits[name])
    else
        Â = sx.split_A
        b̂ = sx.split_b
    end
        
    if use_zono
        zonos = Vector{Zonotope}()
        for i in 1:dim(sx.z)
            zono = Zonotope([sx.z.center[i], sy.z.center[i]], [sx.z.generators[i,:]'; sy.z.generators[i,:]'])
            push!(zonos, zono)
        end
        # TODO: add method=method ???
        relaxations = get_relaxation_σy_zono.(zonos; n_samples=100, max_steps=10000, optimality_gap=1e-8, printing=false)
    else
        relaxations = get_relaxation_σ_y.(lx, ux, ly, uy, n_samples=n_samples, method=method)
    end
    M = reduce(hcat, [collect(r) for r in relaxations])'

    ĉ = M[:,1] .* x.center .+ M[:,2] .* y.center .+ M[:,3]
    Ĝ = M[:,1] .* x.generators .+ M[:,2] .* y.generators
    # append new generators, need new generator for every dim as non-linear.
    Ĝ = [Ĝ I(dim(x)) .* M[:,4]]

    ẑ = Zonotope(ĉ, Ĝ)

    # sx and sy should have the same splits!
    @assert sx.splits == sy.splits
    # sx and sy should have the same generators!
    @assert sx.generator_map == sy.generator_map

    # TODO: real split_A and split_b
    layer_importance = vec(sum(abs.(x.generators) .+ abs.(y.generators), dims=1))
    importance = [sx.importance .+ sy.importance .+ layer_importance; zeros(dim(x))]
    sz = SplitZonotope(ẑ, sx.splits, sx.bounds, copy(sx.generator_map), Â, b̂, sx.shape, importance)
    push!(sz.generator_map, [(name, i) for i in 1:dim(x)]...)
    return sz
end


function propagate_σ_tanh(sx::SplitZonotope, sy::SplitZonotope, name::String; n_samples=100, use_zono=false, method=:remezlike)
    x = sx.z
    y = sy.z

    lx, ux, ly, uy = get_bounds!(sx, sy, name)

    # splits should be the same in both!
    if name in keys(sx.splits)
        #idx1, idx2, l_split, u_split = unzip(sx.splits[name])
        lx, ux, ly, uy, Â, b̂ = get_split_bounds(sx, sy, sx.splits[name])
    else
        Â = sx.split_A
        b̂ = sx.split_b
    end
    
    if use_zono
        zonos = Vector{Zonotope}()
        for i in 1:dim(sx.z)
            zono = Zonotope([sx.z.center[i], sy.z.center[i]], [sx.z.generators[i,:]'; sy.z.generators[i,:]'])
            push!(zonos, zono)
        end
        # TODO: add method=method
        relaxations = get_relaxation_σtanh_zono.(zonos; n_samples=100, max_steps=10000, optimality_gap=1e-8, printing=false)
    else
        relaxations = get_relaxation_σ_tanh.(lx, ux, ly, uy, n_samples=n_samples, method=method)
    end
    M = reduce(hcat, [collect(r) for r in relaxations])'

    ĉ = M[:,1] .* x.center .+ M[:,2] .* y.center .+ M[:,3]
    Ĝ = M[:,1] .* x.generators .+ M[:,2] .* y.generators
    # append new generators, need new generator for every dim as non-linear.
    Ĝ = [Ĝ I(dim(x)) .* M[:,4]]

    ẑ = Zonotope(ĉ, Ĝ)

    # sx and sy should have the same splits!
    @assert sx.splits == sy.splits
    # sx and sy should have the same generators!
    @assert sx.generator_map == sy.generator_map
    # TODO: real split_A and split_b
    layer_importance = vec(sum(abs.(x.generators) .+ abs.(y.generators), dims=1))
    importance = [sx.importance .+ sy.importance .+ layer_importance; zeros(dim(x))]
    sz = SplitZonotope(ẑ, sx.splits, sx.bounds, copy(sx.generator_map), Â, b̂, sx.shape, importance)
    push!(sz.generator_map, [(name, i) for i in 1:dim(x)]...)

    @assert length(sz.importance) == length(sz.generator_map) "Did you reuse an old zonotope? (The generator_map is not cleared when a new propagation starts.)"
    return sz
end