

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

        idx = findfirst(x -> x == idx, idxs)
        if !isnothing(idx)
            lx = max(lx, lxs[idx])
            ux = min(ux, uxs[idx])
            ly = max(ly, lys[idx])
            uy = min(uy, uys[idx])
        end
    end

    return lx, ux, ly, uy   
end


"""
Split two inputs of LSTM activation function into four quadrants.
"""
function split_lstm_layer(sz::SplitZonotope, input_shape, split_layer, split_idx)
    lx, ux, ly, uy = get_split_bounds(sz.splits, sz.bounds, split_layer, split_idx)
    mx = 0.5 * (lx + ux)
    my = 0.5 * (ly + uy)

    l, u = sz.bounds["input"]
    input_set = Hyperrectangle(low=l, high=u)
    zs = [SplitZonotope(input_set, input_shape) for i in 1:4]

    new_splits = [(split_idx, lx, mx, ly, my), (split_idx, lx, mx, my, uy),
                (split_idx, mx, ux, ly, my), (split_idx, mx, ux, my, uy)]

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


function split_split_zonotope(sz::SplitZonotope, input_shape)
    # need ...[1,:] since importance would be (1,n_gens) matrix, but need vector idx
    importance = sum(abs.(sz.z.generators), dims=1)[1,:]
    split_layer, split_idx = sz.generator_map[argmax(importance)]

    if startswith(split_layer, "input")
        zs, new_splits = split_input(sz, input_shape, split_layer, split_idx)
    elseif startswith(split_layer, "relu")
        zs, new_splits = split_relu_layer(sz, input_shape, split_layer, split_idx)
    elseif startswith(split_layer, "LSTM")
        zs, new_splits = split_lstm_layer(sz, input_shape, split_layer, split_idx)
    else
        throw(ArgumentError("Splitting layer $(split_layer) not supported!"))
    end
    
    # make sure children have distinct dictionaries for bounds and splits!
    for z in zs
        for (k, v) in sz.splits
            z.splits[k] = copy(v)
        end

        for (k, v) in sz.bounds
            # TODO: is this efficient???
            z.bounds[k] = deepcopy(v)
        end
    end

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

    return zs  
end