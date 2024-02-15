

"""
Builds LP for optimization over space described by SplitZonotope.

Maximizes output opt_idx of the SplitZonotope subject to the split constraints stored.

args:
    sz - SplitZonotope whose output should be maximized

kwargs:
    opt - the optimizer for the LP
    opt_idx - the dimension of the SplitZonotope to maximize
"""
function build_model(sz::SplitZonotope; opt=Gurobi.Optimizer, opt_idx=1, silent=true, maximize=true)
    z = sz.z
    m, n = size(sz.split_A)

    model = Model(opt)
    silent && set_silent(model)
    @variable(model, -1 <= x[1:ngens(z)] <= 1)

    if maximize
        @objective(model, Max, z.generators[opt_idx,:]' * x + z.center[opt_idx])
    else
        @objective(model, Min, z.generators[opt_idx,:]' * x + z.center[opt_idx])
    end

    A = [sz.split_A zeros(m, ngens(z) - n)]
    @constraint(model, split_constraint, A*x .<= sz.split_b)
    return model
end


"""
Calculates concrete bounds on the values overapproximated by the SplitZonotope.

Bounds can be calculated either by box-overapproximation (which are exact bounds, if there are no split constraints)
or via solving an LP for every value overapproximated by the SplitZonotope.

args:
    sz - the SplitZonotope to calculate the bounds for

kwargs:
    skip_lp - whether to skip the LP and just use quick box overapproximation
    upper - if true, calculate upper bounds, if false calculate lower bounds
    opt - the optimizer to use (has to be a callable with zero arguments returning 
          a JuMP Optimizer constructor e.g. () -> Gurobi.Optimizer())
    
returns:
    vector of concrete bounds
"""
function bounds(sz::SplitZonotope; skip_lp=false, upper=true, opt=() -> Gurobi.Optimizer(GRB_ENV[]))
    if skip_lp
        H = overapproximate(sz.z, Hyperrectangle)
        bds = upper ? high(H) : low(H)
    else
        # only for flat vectors for now, ignore batch dim
        n_out = sz.shape[1]
        bds = zeros(n_out)
        for i in 1:n_out
            model = build_model(sz, opt_idx=i, maximize=upper, opt=opt)
            optimize!(model)

            if termination_status(model) == OPTIMAL
                bds[i] = objective_value(model)
            elseif termination_status(model) == INFEASIBLE
                # if we want an upper bound, the smallest upper bound to an infeasible problem is -Inf
                # reverse for if we want a lower bound
                bds[i] = upper ? -Inf : Inf
            else
                throw(InvalidStateException("The LP was not infeasible and we did not find an optimal solution!", :not_infeasible_not_optimal))
            end
        end
    end

    return bds
end


# tries to prove that output y of the network is in polytope, i.e. Ay - b ≤ 0 -> minimizes error maximum(Ay - b)
function contained_within_polytope_sz(nn::CompGraph, input_set::AbstractHyperrectangle, polytope, params::PriorityOptimizerParameters;
    solver=DPNFV(), split=split_largest_interval, concrete_sample=:Center)
    A, b = tosimplehrep(polytope)
    nn_spec = merge_into_network(nn, A, b)

    in_shape = map(x -> ifelse(isa(x, Integer), x, 1), nn_spec.input_shape)
    s = SplitZonotope(input_set, in_shape)

    function approximate_optimize_cell(cell)   
        out_cell = propagate(solver, nn_spec, cell)
        violations = bounds(out_cell, upper=true)
        max_violation = maximum(violations)

        return max_violation, out_cell
    end

    achievable_value = cell -> begin
        l, u = cell.bounds["input"]
        center = 0.5 .* (l .+ u)
        y = maximum(propagate(nn_spec, reshape(center, nn_spec.input_shape)))
        return center, y
    end

    return general_priority_optimization(s, approximate_optimize_cell, achievable_value, params, true,
            split=split, bound_threshold_realizable=0., bound_threshold_approximate=0.)
end