

struct LSTMSolver <: Solver end


function forward_node(solver::LSTMSolver, L::Linear, sz::SplitZonotope)
    # TODO: use L.dense, but somehow ignore the bias?
    # TODO: figure out way for Float32/Float64 conversion
    
    # CAN'T use LazySets.affine_map since, it only yields **one** generator (i.e. summarizes the generators)
    # when the result is only one-dimensional (see https://github.com/JuliaReach/LazySets.jl/blob/ded315e296ab240a54176f07477343cdfe3f95b1/src/Interfaces/AbstractZonotope.jl#L288)
    # ẑ = LazySets.affine_map(Float64.(L.dense.weight), sz.z, Float64.(L.dense.bias))

    # therefore, we do it ourself:
    Ĝ = L.dense.weight * sz.z.generators
    ĉ = L.dense.weight * sz.z.center .+ L.dense.bias
    ẑ = Zonotope(ĉ, Ĝ)

    ĉ = L.dense(reshape(sz.z.center, sz.shape))
    shape = size(ĉ)

    return SplitZonotope(ẑ, sz, shape)
end


function forward_node(solver::LSTMSolver, L::AddConst, sz::SplitZonotope)
    return direct_sum(sz, L.c)
end


function forward_node(solver::LSTMSolver, L::Convolution, sz::SplitZonotope)
    c = reshape(sz.z.center, sz.shape)
    ĉ = L.conv(c)

    # put equations into batch dimension
    if (sz.shape[end] == 1) && length(sz.shape) > 1
        G = reshape(sz.z.generators, (sz.shape[1:end-1]..., :))
    else
        G = reshape(sz.z.generators, (sz.shape..., :))
    end

    # need to handle conv ourselves, because we don't want to add bias to generator matrix
    # see https://github.com/FluxML/Flux.jl/blob/f4b47611cb731b41879a0af10439026a67c942e1/src/layers/conv.jl#L197-L221
    Ĝ = Flux.conv(G, L.conv.weight, Flux.conv_dims(L.conv, G))

    shape = size(ĉ)

    ẑ = Zonotope(Float64.(vec(ĉ)), length(Ĝ) > 0 ? Float64.(reshape(Ĝ, (:, size(Ĝ)[end]))) : Matrix{Float64}(undef, length(ĉ), 0))
    return SplitZonotope(ẑ, sz, shape)
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


function forward_node(solver::LSTMSolver, lstm_cell::LSTMCell, (i, sh, sc)::Union{Tuple{Integer, SplitZonotope, SplitZonotope},Tuple{Integer, AbstractArray, AbstractArray}}, 
                      sx::SplitZonotope; n_samples=100)
    hs4, input_size = size(lstm_cell.linear_ih.dense.weight)
    hidden_size = floor(Integer, hs4 / 4)
        

    g_ih = forward_node(solver, lstm_cell.linear_ih, sx)
    g_hh = forward_node(solver, lstm_cell.linear_hh, sh)

    g = direct_sum(g_ih, g_hh)

    z = g.z
    z_i = z[1:hidden_size]
    z_f = z[  hidden_size+1:2*hidden_size]
    z_c = z[2*hidden_size+1:3*hidden_size]
    z_o = z[3*hidden_size+1:4*hidden_size]

    # TODO: fix shape after indexing!!!
    g_in = SplitZonotope(z_i, g, (hidden_size, g.shape[2:end]...))
    g_f  = SplitZonotope(z_f, g, (hidden_size, g.shape[2:end]...))
    g_c  = SplitZonotope(z_c, g, (hidden_size, g.shape[2:end]...))
    g_o  = SplitZonotope(z_o, g, (hidden_size, g.shape[2:end]...))
    
    # i is the loop iteration of the lstm cell 
    # (lstm_name_5_σy_1, 4)  
    # --> generator corresponding to 5th lstm unrolling 
    # --> for 1st σ(x)*y non-linearity
    # --> for 4th neuron in that σ(x)*y layer 
    ĝ_f, sĉ = expand_generators(g_f, sc)
    fc = propagate_σ_y(ĝ_f, sĉ, lstm_cell.name * "_$(i)_σy_1", n_samples=n_samples)
    ic = propagate_σ_tanh(g_in, g_c, lstm_cell.name * "_$(i)_σtanh_1", n_samples=n_samples)
    ĉ = direct_sum(fc, ic)

    eĉ, ĝ_o = expand_generators(ĉ, g_o)
    ĥ = propagate_σ_tanh(ĝ_o, eĉ, lstm_cell.name * "_$(i)_σtanh_2", n_samples=n_samples)
    
    return ĥ, ĉ
end


function forward_node(solver::LSTMSolver, lstm_layer::LSTMLayer, sx::SplitZonotope; n_samples=100)
    # TODO: cleaner lstm_cell/flux_cell construct?
    lstm_cell = extract_cell(lstm_layer)
    flux_cell = lstm_layer.cell
    # counter for unrolling and initial state
    state = (0, flux_cell.state0...)

    # last dimension is length of the sequence
    timesteps = sx.shape[end]
    for i in 1:timesteps
        # shape is (features, batch, sequence_length)
        sz_lstm = get_tensor_idx(sx, :, :, i)

        h, c = forward_node(solver, lstm_cell, state, sz_lstm, n_samples=n_samples)
        state = (i+1, h, c)
    end

    n_steps, h, c = state
    return h, c 
end