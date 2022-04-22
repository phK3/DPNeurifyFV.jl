
# TODO: needed to rename solver to DPNFV, because DPNEurifyFV is already the module name.
#       can we find a better solution?
@with_kw struct DPNFV <: Solver
    # :NeurifyRelax uses lower and upper relaxation of Neurify for Relu
    # :DeepPolyRelax uses 0/1-lower bound gradient of DeepPoly
    method = :NeurifyRelax
    max_vars = 10
    var_frac = 0.5
    get_fresh_var_idxs = get_fresh_var_idxs_largest_range
end


function NV.forward_linear(solver::DPNFV, L::LayerNegPosIdx, input::SymbolicIntervalFVHeur)
    Low, Up = interval_map(L.W_neg, L.W_pos, input.Low, input.Up)
    Low[:, end] .+= L.bias
    Up[:, end] .+= L.bias
    return init_symbolic_interval_fvheur(input, Low, Up)
end


function NV.forward_act(solver::DPNFV, L::LayerNegPosIdx{ReLU}, input::SymbolicIntervalFVHeur)
    n_node = n_nodes(L)
    n_sym = get_n_sym(input)
    n_in = get_n_in(input)
    current_n_vars = get_n_vars(input)

    subs_Low_Low, subs_Up_Up = substitute_variables(input.Low, input.Up, input.var_los, input.var_his, n_in, current_n_vars)
    # substitute_variables treats its first arguments as LB and the second as UB, thus we get the LB of input.Up
    # and the UB of input.Low
    subs_Up_Low, subs_Low_Up = substitute_variables(input.Up, input.Low, input.var_los, input.var_his, n_in, current_n_vars)

    low_low = lower_bounds(subs_Low_Low, domain(input), input.lbs[L.index], input.ubs[L.index])
    low_up = upper_bounds(subs_Low_Up, domain(input), low_low, input.ubs[L.index])

    up_up = upper_bounds(subs_Up_Up, domain(input), low_up, input.ubs[L.index])
    up_low = lower_bounds(subs_Up_Low, domain(input), low_low, up_up)

    ### calculate importance score
    crossing = is_crossing.(low_low, up_up)
    layer_importance = sum(abs.(subs_Low_Low[crossing, :]), dims=1) .+ sum(abs.(subs_Up_Up[crossing, :]), dims=1)
    importance = input.importance .+ layer_importance[1:end-1]  # constant term doesn't need importance

    if L.index >= input.n_layers - 1
        # introducing fresh variables in the next to last layer doesn't improve
        # bounds, as there is no opportunity for them to cancel out
        fv_idxs = []
        n_vars = 0
    else
        fv_idxs = solver.get_fresh_var_idxs(input.max_vars, current_n_vars, low_low, up_up, solver.var_frac)
        n_vars = length(fv_idxs)
    end

    if solver.method == :OuterBounds
        λ_l = relaxed_relu_gradient_lower.(low_low, up_up)
        λ_u = relaxed_relu_gradient.(low_low, up_up)
        β_u = -low_low
    elseif solver.method == :NeurifyRelax
        λ_l = relaxed_relu_gradient.(low_low, low_up)
        λ_u = relaxed_relu_gradient.(up_low, up_up)
        β_u = -up_low
    elseif solver.method == :DeepPolyRelax
        λ_l = relaxed_relu_gradient_lower.(low_low, low_up)
        λ_u = relaxed_relu_gradient.(up_low, up_up)
        β_u = -up_low
    end

    ### apply relaxation to symbolic bounds
    output_Up = λ_u .* input.Up
    output_Up[:, end] .+= λ_u .* max.(β_u, 0)

    output_Low = λ_l .* input.Low

    ### apply relaxation to substituted bounds
    subs_Up_Up = λ_u .* subs_Up_Up
    subs_Up_Up[:, end] .+= λ_u .* max.(β_u, 0)

    subs_Low_Low = λ_l .* subs_Low_Low


    ### introduce new symbolic variables
    ## TODO: preserve **type** of symbolic interval through zeros(T, n_node, n_vars)
    if n_vars > 0
        output_Low = [output_Low[:, 1:n_sym] zeros(n_node, n_vars) output_Low[:, end]]
        output_Up = [output_Up[:, 1:n_sym] zeros(n_node, n_vars) output_Up[:, end]]
    end

    for (i, v) in enumerate(fv_idxs)
        # store symbolic bounds on fresh variables
        input.var_los[current_n_vars + i, :] .= subs_Low_Low[v, :]
        input.var_his[current_n_vars + i, :] .= subs_Up_Up[v, :]

        # set corresponding entry to unit-vec
        output_Low[v,:] .= unit_vec(n_sym + i, n_sym + 1 + n_vars)
        output_Up[v,:]  .= unit_vec(n_sym + i, n_sym + 1 + n_vars)
    end

    output = SymbolicIntervalFVHeur(output_Low, output_Up, input.domain, input.lbs, input.ubs,
                                    input.var_los, input.var_his,
                                    input.max_vars, importance, input.n_layers)
    output.lbs[L.index] .= low_low
    output.ubs[L.index] .= up_up

    return output
end


function NV.forward_act(solver::DPNFV, L::LayerNegPosIdx{Id}, input::SymbolicIntervalFVHeur)
    n_node = n_nodes(L)
    n_sym = get_n_sym(input)
    n_in = get_n_in(input)
    current_n_vars = get_n_vars(input)

    subs_Low, subs_Up = substitute_variables(input.Low, input.Up, input.var_los, input.var_his, n_in, current_n_vars)

    low_low = lower_bounds(subs_Low, domain(input), input.lbs[L.index], input.ubs[L.index])
    up_up = upper_bounds(subs_Up, domain(input), low_low, input.ubs[L.index])
    input.lbs[L.index] .= low_low
    input.ubs[L.index] .= up_up

    return input
end
