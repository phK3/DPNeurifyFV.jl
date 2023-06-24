

function forward_node(solver::DPNFV, L::Linear, s::SymbolicIntervalGraph)
    Low = L.dense⁺(s.Low) .+ L.dense⁻(s.Up)
    Up  = L.dense⁺(s.Up)  .+ L.dense⁻(s.Low)
    
    # add bias to last entry in batch dimension
    selectdim(Low, ndims(Low), size(Low, ndims(Low))) .+= L.dense.bias
    selectdim(Up,  ndims(Up),  size(Up,  ndims(Up)))  .+= L.dense.bias

    return init_symbolic_interval_graph(s, Low, Up)
end


function forward_node(solver::DPNFV, L::Relu, s::SymbolicIntervalGraph)
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
end

