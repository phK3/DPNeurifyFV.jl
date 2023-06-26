

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


function forward_node(solver::DPNFV, L::Sub, s₁::SymbolicIntervalGraph, s₂::SymbolicIntervalGraph)
    Low = s₁.Low .- s₂.Up
    Up  = s₁.Up  .- s₂.Low
    return init_symbolic_interval_graph(s, Low, Up)
end


function forward_node(solver::DPNFV, L::Sub, s::SymbolicIntervalGraph, x::AbstractArray)
    return add_constant(s, .-x)
end


function forward_node(solver::DPNFV, L::Convolution, s::SymbolicIntervalGraph)
    Low = L.conv⁺(s.Low) .+ L.conv⁻(s.Up)
    Up  = L.conv⁺(s.Up)  .+ L.conv⁻(s.Low)

    # add bias to last entry in batch dimension
    add_constant!(Low, L.conv.bias)
    add_constant!(Up,  L.conv.bias)

    return init_symbolic_interval_graph(s, Low, Up)
end


function forward_node(solver::DPNFV, L::ConvolutionTranspose, s::SymbolicIntervalGraph)
    Low = L.convt⁺(s.Low) .+ L.convt⁻(s.Up)
    Up  = L.convt⁺(s.Up)  .+ L.convt⁻(s.Low)

    # add bias to last entry in batch dimension
    add_constant!(Low, L.convt.bias)
    add_constant!(Up,  L.convt.bias)

    return init_symbolic_interval_graph(s, Low, Up)
end


function forward_node(solver, L::BatchNormalization, s::SymbolicIntervalGraph)
    # subtract mean from last entry in batch dimension
    ŝ = add_constant(s, .- L.batchnorm.μ)

    Low = L.batchnorm⁺(ŝ.Low) .+ L.batchnorm⁻(ŝ.Up)
    Up  = L.batchnorm⁺(ŝ.Up)  .+ L.batchnorm⁻(ŝ.Low)

    add_constant!(Low, L.batchnorm.β)
    add_constant!(Up,  L.batchnorm.β)

    return init_symbolic_interval_graph(s, Low, Up)
end


function forward_node(solver::DPNFV, L::Reshape, s::SymbolicIntervalGraph)
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

