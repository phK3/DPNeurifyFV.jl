

function forward_node(solver::DPNFV, L::Linear, s::SymbolicIntervalGraph)
    Low = L.dense⁺(s.Low) .+ L.dense⁻(s.Up)
    Up  = L.dense⁺(s.Up)  .+ L.dense⁻(s.Low)
    
    # add bias to last entry in batch dimension
    add_constant!(Low, L.dense.bias)
    add_constant!(Up,  L.dense.bias)
   
    return init_symbolic_interval_graph(s, Low, Up)
end


function forward_node(solver::DPNFV, L::Add, s₁::SymbolicIntervalGraph, s₂::SymbolicIntervalGraph)
    # TODO: check fresh variables
    Low = s₁.Low .+ s₂.Low
    Up  = s₁.Up.+ s₂.Up
    return init_symbolic_interval_graph(s, Low, Up)
end


function forward_node(solver::DPNFV, L::Add, s::SymbolicIntervalGraph, x::AbstractArray)
    return add_constant(s, x)
end


function forward_node(solver::DPNFV, L::Add, x::AbstractArray, s::SymbolicIntervalGraph)
    return forward_node(solver, L, s, x)
end


function forward_node(solver::DPNFV, L::AddConst, s::SymbolicIntervalGraph)
    return add_constant(s, L.c)
end


function forward_node(solver::DPNFV, L::Sub, s₁::SymbolicIntervalGraph, s₂::SymbolicIntervalGraph)
    Low = s₁.Low .- s₂.Up
    Up  = s₁.Up  .- s₂.Low
    return init_symbolic_interval_graph(s, Low, Up)
end


function forward_node(solver::DPNFV, L::Sub, s::SymbolicIntervalGraph, x::AbstractArray)
    return add_constant(s, .-x)
end


function forward_node(solver::DPNFV, L::SubConst, s::SymbolicIntervalGraph)
    # want L.c - s = -s + L.c
    add_constant(init_symbolic_interval_graph(s, .-s.Low, .-s.Up), L.c)
    return add_constant(s, L.c)
end


function forward_node(solver::DPNFV, L::Convolution, s::SymbolicIntervalGraph)
    Low = L.conv⁺(s.Low) .+ L.conv⁻(s.Up)
    Up  = L.conv⁺(s.Up)  .+ L.conv⁻(s.Low)

    # add bias to last entry in batch dimension
    add_constant!(Low, Flux.conv_reshape_bias(L.conv))
    add_constant!(Up,  Flux.conv_reshape_bias(L.conv))

    return init_symbolic_interval_graph(s, Low, Up)
end


function forward_node(solver::DPNFV, L::ConvolutionTranspose, s::SymbolicIntervalGraph)
    Low = L.convt⁺(s.Low) .+ L.convt⁻(s.Up)
    Up  = L.convt⁺(s.Up)  .+ L.convt⁻(s.Low)

    # add bias to last entry in batch dimension
    add_constant!(Low, Flux.conv_reshape_bias(L.convt))
    add_constant!(Up,  Flux.conv_reshape_bias(L.convt))

    return init_symbolic_interval_graph(s, Low, Up)
end


function forward_node(solver, L::BatchNormalization, s::SymbolicIntervalGraph)
    N = ndims(s)
    stats_shape = ntuple(i -> i == N-1 ? size(s, N-1) : 1, N)
    μ = reshape(L.batchnorm.μ, stats_shape)
    β = reshape(L.batchnorm.β, stats_shape)
    
    # subtract mean from last entry in batch dimension
    ŝ = add_constant(s, .- μ)

    Low = L.batchnorm⁺(ŝ.Low) .+ L.batchnorm⁻(ŝ.Up)
    Up  = L.batchnorm⁺(ŝ.Up)  .+ L.batchnorm⁻(ŝ.Low)

    add_constant!(Low, β)
    add_constant!(Up,  β)

    return init_symbolic_interval_graph(s, Low, Up)
end


#= function forward_node(solver::DPNFV, L::Reshape, s::SymbolicIntervalGraph)
    shape = [L.shape...]

    # if batch-dim already included in L.shape?
    if length(shape) == ndims(s.Low)
        # batch-dim included
        @assert shape[end] == 1 "Reshaping the batch-dimension is not allowed! (at node $(L.name))"
        shape[end] .= size(s.low, ndims(s.Low))
    else
        # batch-dim not included
        push!(shape, size(s.Low, ndims(s.Low)))
    end

    Low = reshape(s.Low, shape...)
    Up  = reshape(s.Up,  shape...)

    return init_symbolic_interval_graph(s, Low, Up)
end =#


#= function forward_node(solver::DPNFV, L::Relu, s::SymbolicIntervalGraph)
    # TODO: consider fresh variables, right now only for vanilla symbolic interval
    # TODO: consider stored bounds (and also disabling of storing bounds!)
    Low = Flux.flatten(s.Low)
    Up  = Flux.flatten(s.Up)
    n_neurons = size(Low, 1)

    lbs = fill(-Inf, n_neurons)
    ubs = fill(Inf, n_neurons)
    ll = lower_bounds(Low, domain(s), lbs, ubs)
    lu = upper_bounds(Low, domain(s), lbs, ubs)
    ul = lower_bounds(Up,  domain(s), lbs, ubs)
    uu = upper_bounds(Up,  domain(s), lbs, ubs)

    crossing = is_crossing.(ll, uu)
    layer_importance = sum(abs.(Low[crossing, :]), dims=1) .+ sum(abs.(Up[crossing, :]), dims=1)
    importance = s.importance .+ layer_importance[1:end-1]  # constant term doesn't need importance

    if solver.method == :OuterBounds
        λ_l = relaxed_relu_gradient_lower.(ll, uu)
        λ_u = relaxed_relu_gradient.(ll, uu)
        β_u = -ll
    elseif solver.method == :NeurifyRelax
        λ_l = relaxed_relu_gradient.(ll, lu)
        λ_u = relaxed_relu_gradient.(ul, uu)
        β_u = -ul
    elseif solver.method == :DeepPolyRelax
        λ_l = relaxed_relu_gradient_lower.(ll, lu)
        λ_u = relaxed_relu_gradient.(ul, uu)
        β_u = -ul
    end

    L̂ = λ_l .* Low
    Û = λ_u .* Up
    Û[:, end] .+= λ_u .* max.(0, β_u)

    # get in right shape after flattening
    L = reshape(L̂, size(s.Low))
    U = reshape(Û, size(s.Up))


    output = SymbolicIntervalGraph(L, U, domain(s), s.lbs, s.ubs, 
                                   s.var_los, s.var_his, s.var_ids, 
                                   s.max_vars, importance)

    return output
end =#

function forward_node(solver::DPNFV, L::Relu, s::SymbolicIntervalGraph)
    # TODO: consider fresh variables, right now only for vanilla symbolic interval
    # TODO: consider stored bounds (and also disabling of storing bounds!)
    Low = Flux.flatten(s.Low)
    Up  = Flux.flatten(s.Up)
    n_neurons = size(Low, 1)
    n_in = get_n_in(s)
    n_sym = get_n_sym(s)
    current_n_vars = get_n_vars(s)

    subs_LL, subs_UU = substitute_variables(s.Low, s.Up, s.var_los, s.var_his, n_in, current_n_vars)
    subs_UL, subs_LU = substitute_variables(s.Up, s.Low, s.var_los, s.var_his, n_in, current_n_vars)

    lbs = fill(-Inf, n_neurons)
    ubs = fill(Inf, n_neurons)
    ll = lower_bounds(subs_LL, domain(s), lbs, ubs)
    lu = upper_bounds(subs_LU, domain(s), lbs, ubs)
    ul = lower_bounds(subs_UL,  domain(s), lbs, ubs)
    uu = upper_bounds(subs_UU,  domain(s), lbs, ubs)

    crossing = is_crossing.(ll, uu)
    layer_importance = sum(abs.(subs_LL[crossing, :]), dims=1) .+ sum(abs.(subs_UU[crossing, :]), dims=1)
    importance = s.importance .+ layer_importance[1:end-1]  # constant term doesn't need importance

    fv_idxs = solver.get_fresh_var_idxs(s.max_vars, current_n_vars, ll, uu, solver.var_frac)
    n_vars = length(fv_idxs)

    if solver.method == :OuterBounds
        λ_l = relaxed_relu_gradient_lower.(ll, uu)
        λ_u = relaxed_relu_gradient.(ll, uu)
        β_u = -ll
    elseif solver.method == :NeurifyRelax
        λ_l = relaxed_relu_gradient.(ll, lu)
        λ_u = relaxed_relu_gradient.(ul, uu)
        β_u = -ul
    elseif solver.method == :DeepPolyRelax
        λ_l = relaxed_relu_gradient_lower.(ll, lu)
        λ_u = relaxed_relu_gradient.(ul, uu)
        β_u = -ul
    end

    # relaxation of symbolic bounds
    L̂ = λ_l .* Low
    Û = λ_u .* Up
    Û[:, end] .+= λ_u .* max.(0, β_u)

    # relaxation of substituted bounds
    subs_LL = λ_l .* subs_LL
    subs_UU = λ_u .* subs_UU
    subs_UU[:, end] .+= λ_u .* max.(β_u, 0)

    # introduce new symbolic variables
    if n_vars > 0
        L̂ = [L̂[:, 1:n_sym] zeros(n_neurons, n_vars) L̂[:, end]]
        Û = [Û[:, 1:n_sym] zeros(n_neurons, n_vars) Û[:, end]]
    end

    for (i, v) in enumerate(fv_idxs)
        # store symbolic bounds on fresh variables
        s.var_los[current_n_vars + i, :] .= subs_LL[v, :]
        s.var_his[current_n_vars + i, :] .= subs_UU[v, :]

        # set corresponding entry to unit vec
        L̂[v, :] .= unit_vec(n_sym + i, n_sym + 1 + n_vars)
        Û[v, :] .= unit_vec(n_sym + i, n_sym + 1 + n_vars)
    end

    # get in right shape after flattening
    # batch dimension can get larger as there are now more coefficients for the
    # fresh variables
    L = reshape(L̂, size(s.Low)[1:end-1]..., :)
    U = reshape(Û, size(s.Up)[1:end-1]..., :)


    output = SymbolicIntervalGraph(L, U, domain(s), s.lbs, s.ubs, 
                                   s.var_los, s.var_his, s.var_ids, 
                                   s.max_vars, importance)

    return output
end


function forward_node(solver::DPNFV, L::Gather, s::SymbolicIntervalGraph)
    # TODO: what to do about number of fresh variables?
    Low = my_gather(s.Low, L.inds, L.axis)
    Up  = my_gather(s.Up,  L.inds, L.axis)

    return init_symbolic_interval_graph(s, Low, Up)
end

