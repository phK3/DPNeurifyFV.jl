

"""
Extract better of lower and upper bounds stored and bounds stored in splits for the specified
neuron in the specified layer.

args:
    splits - Dictionary of splits (for the whole network)
    bounds - dictionary of bounds (for the whole network)
    layer_name - name of the layer of interest
    idx - neuron index in that layer

returns:
    lx, ux, ly, uy - best bounds for the two inputs of that neuron.
"""
function get_split_bounds(splits::Dict, bounds::Dict, layer_name::String, idx::Integer)
    lx, ux, ly, uy = (b[idx] for b in bounds[layer_name])
    
    if haskey(splits, layer_name)
        idxs, lxs, uxs, lys, uys = unzip(splits[layer_name])

        if !isnothing(idx)
            lx = max(lx, maximum(lxs[idxs .== idx], init=-Inf))
            ux = min(ux, minimum(uxs[idxs .== idx], init=Inf))
            ly = max(ly, maximum(lys[idxs .== idx], init=-Inf))
            uy = min(uy, minimum(uys[idxs .== idx], init=Inf))
        end
    end

    return lx, ux, ly, uy   
end


function split_bivariate_optimal(f_overapprox, lx, ux, ly, uy; n_test=10)
    x_best = 0.5*(lx + ux)
    ϵx = Inf
    mxs = range(lx, ux, length=n_test+2)[2:end-1]
    for mx in mxs
        a₁, b₁, c₁, ϵ₁ = f_overapprox(lx, mx, ly, uy)
        a₂, b₂, c₂, ϵ₂ = f_overapprox(mx, ux, ly, uy)

        ϵ = max(abs(ϵ₁), abs(ϵ₂))
        if ϵ < ϵx
            ϵx = ϵ
            x_best = mx
        end
    end

    y_best = 0.5*(ly + uy)
    ϵy = Inf
    mys = range(ly, uy, length=n_test+2)[2:end-1]
    for my in mys
        a₁, b₁, c₁, ϵ₁ = f_overapprox(lx, ux, ly, my)
        a₂, b₂, c₂, ϵ₂ = f_overapprox(lx, ux, my, uy)

        ϵ = max(abs(ϵ₁), abs(ϵ₂))
        if ϵ < ϵx
            ϵy = ϵ
            y_best = my
        end
    end

    if ϵx < ϵy
        new_splits = [(lx, x_best, ly, uy), (x_best, ux, ly, uy)]
    else
        new_splits = [(lx, ux, ly, y_best), (lx, ux, y_best, uy)]
    end

    return new_splits
end


"""
Split two inputs of LSTM activation function into four quadrants.

args
    sz - the SplitZonotope that is the output from the current overapproximation
    input_shape - the input shape of the neural network
    split_layer - the name of the layer where the split is located
    split_idx - the index of the neuron in the split_layer to be split

kwargs:
    split_method - different methods to subdivide the current input domain of the split neuron
"""
function split_lstm_layer(sz::SplitZonotope, input_shape, split_layer, split_idx; split_method=:zero)
    lx, ux, ly, uy = get_split_bounds(sz.splits, sz.bounds, split_layer, split_idx)
    
    # TODO: remove debug output
    #println("layer: ", split_layer, ", idx: ", split_idx, " -> ", (lx, ux), " × ", (ly, uy))

    if split_method == :zero
        # first split at the origin/the axes before splitting the domain into four equal quadrants
        mx = lx < 0 && ux > 0 ? 0. : 0.5 * (lx + ux)
        my = ly < 0 && uy > 0 ? 0. : 0.5 * (ly + uy)

        new_splits = [(split_idx, lx, mx, ly, my), (split_idx, lx, mx, my, uy),
                (split_idx, mx, ux, ly, my), (split_idx, mx, ux, my, uy)]
    elseif split_method == :center
        # split the domain into four equal quadrants
        mx = 0.5 * (lx + ux)
        my = 0.5 * (ly + uy)

        new_splits = [(split_idx, lx, mx, ly, my), (split_idx, lx, mx, my, uy),
                (split_idx, mx, ux, ly, my), (split_idx, mx, ux, my, uy)]
    elseif split_method == :threshold_range
        # as σ(x) ≈ 0 for x ≤ -5 and σ(x) ≈ 1 for x ≥ 5 use this for initial splits
        θ = 3.
        my = ly < 0 && uy > 0 ? 0. : 0.5 * (ly + uy)
        if lx < -θ && ux > θ
            new_splits = [(split_idx, lx, -θ, ly, uy), (split_idx, -θ, θ, ly, my),
                          (split_idx, -θ, θ, my, uy), (split_idx, θ, ux, ly, uy)]
        else
            mx = lx < 0 && ux > 0 ? 0. : 0.5 * (lx + ux)
            new_splits = [(split_idx, lx, mx, ly, my), (split_idx, lx, mx, my, uy),
                    (split_idx, mx, ux, ly, my), (split_idx, mx, ux, my, uy)]
        end
    elseif split_method == :optimal
        if occursin("σy", split_layer)
            splits = split_bivariate_optimal((lx, ux, ly, uy) -> get_relaxation_σ_y(lx, ux, ly, uy, n_samples=100), lx, ux, ly, uy)
        elseif occursin("σtanh", split_layer)
            splits = split_bivariate_optimal((lx, ux, ly, uy) -> get_relaxation_σ_tanh(lx, ux, ly, uy, n_samples=100), lx, ux, ly, uy)
        end

        new_splits = [(split_idx, splits[1]...), (split_idx, splits[2]...)]
    else
        throw(ArgumentError("Splitting method $(split_method) not known!"))
    end

    l, u = sz.bounds["input"]
    input_set = Hyperrectangle(low=l, high=u)
    zs = [SplitZonotope(input_set, input_shape) for i in 1:length(new_splits)]

    return zs, new_splits
end


function split_relu_layer(sz::SplitZonotope, input_shape, split_layer, split_idx)
    l, u = sz.bounds["input"]
    input_set = Hyperrectangle(low=l, high=u)
    
    zs = [SplitZonotope(input_set, input_shape) for i in 1:2]
    new_splits = [(split_idx, -Inf, 0.), (split_idx, 0., Inf)]
    return zs, new_splits
end


function split_input(sz::SplitZonotope, input_shape, split_layer, split_idx)
    l, u = sz.bounds["input"]
    m = 0.5 * (l[split_idx] + u[split_idx])
    
    lm = copy(l)
    lm[split_idx] = m

    um = copy(u)
    um[split_idx] = m

    h₁ = Hyperrectangle(low=l, high=um)
    h₂ = Hyperrectangle(low=lm, high=u)
    zs = [SplitZonotope(h₁, input_shape), SplitZonotope(h₂, input_shape)]
    new_splits = [(split_idx, l[split_idx], m), (split_idx, m, u[split_idx])]

    return zs, new_splits
end


"""
Ensures that the dictionaries for splits and bounds for each SplitZonotope in zs don't point to the same object.

Modifies the dictionaries of the SplitZonotopes in zs!

args:
    sz - the original SplitZonotope
    zs - vector of SplitZonotopes sz was split into
"""
function generate_separate_split_dicts!(sz::SplitZonotope, zs)
    # make sure children have distinct dictionaries for bounds and splits!
    for z in zs
        for (k, v) in sz.splits
            z.splits[k] = copy(v)
        end

        for (k, v) in sz.bounds
            if k != "input"
                # don't use the old bounds for input splitting!!!
                # also input bounds get directly constructed, when the zono is initialized
                # TODO: is this efficient???
                z.bounds[k] = deepcopy(v)
            end
        end
    end
end


"""
Adds the splits to the split dictionaries of the split SplitZonotopes in zs.

Modifies the dictionaries of the SplitZonotopes in zs!

args:
    sz - the original SplitZonotope
    zs - vector of SplitZonotopes sz was split into
    split_layer - the name of the layer where the new splits occured
    new_splits - vector of (neuron_idx, l, u) describing the neuron in the split_layer with new lower and upper bounds on its input.
"""
function add_new_splits!(sz::SplitZonotope, zs, split_layer, new_splits)
     # add new splits
     if haskey(sz.splits, split_layer)
        for (z, new_split) in zip(zs, new_splits)
            push!(z.splits[split_layer], new_split)
        end
    else
        for (z, new_split) in zip(zs, new_splits)
            z.splits[split_layer] = [new_split]
        end
    end
    
end


function split_split_zonotope(sz::SplitZonotope, input_shape; lstm_split_method=:zero)
    # need ...[1,:] since importance would be (1,n_gens) matrix, but need vector idx
    importance = sum(abs.(sz.z.generators), dims=1)[1,:]
function split_split_zonotope_importance(sz::SplitZonotope, input_shape; lstm_split_method=:zero)
    # need ...[1,:] since importance would be (1,n_gens) matrix, but need vector idx
    split_layer, split_idx = sz.generator_map[argmax(sz.importance)]

    return split_split_zonotope(sz, input_shape, split_layer, split_idx, lstm_split_method=lstm_split_method)
end
    split_layer, split_idx = sz.generator_map[argmax(importance)]

    return split_split_zonotope(sz, input_shape, split_layer, split_idx, lstm_split_method=lstm_split_method)
end


function split_split_zonotope(sz::SplitZonotope, input_shape, split_layer, split_idx; lstm_split_method=:zero)
    if startswith(split_layer, "input")
        zs, new_splits = split_input(sz, input_shape, split_layer, split_idx)
    elseif startswith(split_layer, "relu")
        zs, new_splits = split_relu_layer(sz, input_shape, split_layer, split_idx)
    elseif startswith(split_layer, "LSTM")
        zs, new_splits = split_lstm_layer(sz, input_shape, split_layer, split_idx, split_method=lstm_split_method)
    else
        throw(ArgumentError("Splitting layer $(split_layer) not supported!"))
    end
    
    generate_separate_split_dicts!(sz, zs)
    add_new_splits!(sz, zs, split_layer, new_splits)

    return zs  
end


# just a dummy method for now
function split_multiple_times(sz::SplitZonotope, n)
    return [sz]
end