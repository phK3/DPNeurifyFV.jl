

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