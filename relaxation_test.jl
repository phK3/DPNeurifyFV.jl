using DPNeurifyFV, LazySets, PyVnnlib, NeuralVerification, JuMP, LinearAlgebra, Flux, Gurobi, Accessors, Plots, Revise
import VNNLib.NNLoader as NNL

const NV = NeuralVerification
const DP = DPNeurifyFV
const L = DP.LSTMRelaxation


onnx_path = "./networks/lstm_no_initial_state-sim.onnx"
nn = NNL.load_network_dict(DP.CGType, onnx_path)

# change from gather batch-dim to gather seq-dim
gather_node = DP.Gather(["onnx::Gather_103"], ["onnx::Gemm_106"], "Gather_30", fill(-1), 1)
nn.nodes["Gather_30"] = gather_node
for (k, v) in nn.out_dict
      if v.name == "Gather_30"
            nn.out_dict[k] = gather_node
      end
end

nn_logits = DP.CompGraph(nn.nodes, nn.in_node, DP.get_producer(nn, "input.8"), nn.out_dict, nn.input_shape, nn.output_shape)

# define input set
lb = -0.05 .* ones(150)
ub =  0.05 .* ones(150)
input_set = Hyperrectangle(low = lb, high = ub)
sz = DP.SplitZonotope(input_set, nn_logits.input_shape)

ẑ = DP.propagate(DP.LSTMSolver(), nn_logits, sz)


function make_random_lines(n_lines)
    x0s = randn(n_lines)
    x1s = randn(n_lines)
    λls = rand(n_lines)
    λus = min.(1., λls + rand(n_lines))

    return [param for param in zip(x0s, x1s, λls, λus)]
end


function test_line_interval_approx_actfun(domain, f, approx_f; n_test=100, tol=1e-12)
    (x0, x1, y0, y1, λl, λu) = domain
    l, u = approx_f(x0, x1, y0, y1, λl, λu)

    λs = range(λl, λu, n_test)

    xs = (1 .- λs) .* x0 .+ λs .* x1
    ys = (1 .- λs) .* y0 .+ λs .* y1

    fs = f.(xs, ys)

    l_sample = minimum(fs)
    u_sample = maximum(fs)
    
    if l_sample < l - tol || u_sample > u + tol
        if l_sample < l - tol
            println("Minimum must be contained in interval! Got ", l_sample, " < ", l)
        end

        if u_sample > u + tol
            println("Maximum must be contained in interval! Got ", u_sample, " > ", u)
        end

        println("\tline between ", (x0, y0), " and ", (x1, y1), " for λ ∈ ", [λl, λu])
        return false
    end

    return true
end


function test_line_interval_approx_actfun(f, approx_f; n_samples=1000, n_test=100, tol=1e-12)
    x0s = randn(n_samples)
    x1s = randn(n_samples)
    y0s = randn(n_samples)
    y1s = randn(n_samples)
    λls = rand(n_samples)
    λus = min.(1., λls .+ rand(n_samples))

    domains = []
    for (x0, x1, y0, y1, λl, λu) in zip(x0s, x1s, y0s, y1s, λls, λus)
        domain = (x0, x1, y0, y1, λl, λu)
        if !test_line_interval_approx_actfun(domain, f, approx_f, n_test=n_test, tol=tol)
            push!(domains, domain)
        end
    end

    return domains
end


function test_line_interval_approx_σ_y(; n_samples=1000, n_test=100, tol=1e-12)
    f(x,y) = L.σ(x)*y
    approx_f = DP.interval_σy
    test_line_interval_approx_actfun(f, approx_f, n_samples=n_samples, n_test=n_test, tol=tol)
end


function test_line_interval_approx_σ_tanh(; n_samples=1000, n_test=100, tol=1e-12)
    f(x,y) = L.σ(x)*tanh(y)
    approx_f = DP.interval_σtanh
    test_line_interval_approx_actfun(f, approx_f, n_samples=n_samples, n_test=n_test, tol=tol)
end


function test_line_interval_linear_approx_actfun(domain, a, b, c, f, approx_f; n_test=100, tol=1e-12)
    (x0, x1, y0, y1, λl, λu) = domain

    l, u = approx_f(x0, x1, y0, y1, λl, λu, a, b, c)

    λs = range(λl, λu, n_test)

    xs = (1 .- λs) .* x0 .+ λs .* x1
    ys = (1 .- λs) .* y0 .+ λs .* y1

    fs = f.(xs, ys) .- (a .* xs .+ b .* ys .+ c)

    l_sample = minimum(fs)
    u_sample = maximum(fs)

    if l_sample < l - tol || u_sample > u + tol
        if l_sample < l - tol
            println("Minimum must be contained in interval! Got ", l_sample, " < ", l)
        end

        if u_sample > u + tol
            println("Maximum must be contained in interval! Got ", u_sample, " > ", u)
        end

        println("\tline between ", (x0, y0), " and ", (x1, y1), " for λ ∈ ", [λl, λu])
        return false
    end

    return true
end


function test_line_interval_linear_approx_actfun(f, approx_f; n_samples=1000, n_test=100, tol=1e-12)
    x0s = randn(n_samples)
    x1s = randn(n_samples)
    y0s = randn(n_samples)
    y1s = randn(n_samples)
    λls = rand(n_samples)
    λus = min.(1., λls .+ rand(n_samples))

    as = randn(n_samples)
    bs = randn(n_samples)
    cs = randn(n_samples)

    params = []
    for (x0, x1, y0, y1, λl, λu, a, b, c) in zip(x0s, x1s, y0s, y1s, λls, λus, as, bs, cs)
        domain = (x0, x1, y0, y1, λl, λu)
        if !test_line_interval_linear_approx_actfun(domain, a, b, c, f, approx_f, n_test=n_test, tol=tol)
            push!(params, (x0, x1, y0, y1, λl, λu, a, b, c))
        end
    end

    return params
end


function test_line_interval_linear_approx_σ_y(; n_samples=1000, n_test=100, tol=1e-12)
    f(x,y) = L.σ(x)*y
    approx_f = DP.interval_σy_lin
    test_line_interval_linear_approx_actfun(f, approx_f, n_samples=n_samples, n_test=n_test, tol=tol)
end


function test_line_interval_linear_approx_σ_tanh(; n_samples=1000, n_test=100, tol=1e-12)
    f(x,y) = L.σ(x)*tanh(y)
    approx_f = DP.interval_σtanh_lin
    test_line_interval_linear_approx_actfun(f, approx_f, n_samples=n_samples, n_test=n_test, tol=tol)
end


function test_line_interval_overapprox(x0, x1, λl, λu; n_test=100, tol=1e-12)
    lx, ux = DP.line_interval_overapprox(x0, x1, λl, λu)

    λs = range(λl, λu, n_test)
    xs = (1 .- λs) .* x0 .+ λs .* x1

    lx_sample = minimum(xs)
    ux_sample = maximum(xs)

    if lx_sample < lx - tol || ux_sample > ux + tol
        if lx_sample < lx - tol
            println("Minimum must be contained in interval! Got ", lx_sample, " < ", lx)
        end

        if ux_sample > ux + tol
            println("Maximum must be contained in interval! Got ", ux_sample, " > ", ux)
        end

        println("\tline between ", x0, " and ", x1, " for λ ∈ ", [λl, λu])
        return false
    end

    return true
end


function test_line_interval_overapprox(; n_samples=1000, n_test=100, tol=1e-12)
    x0s = randn(n_samples)
    x1s = randn(n_samples)
    λls = rand(n_samples)
    λus = min.(1., λls + rand(n_samples))

    domains = []
    for (x0, x1, λl, λu) in zip(x0s, x1s, λls, λus)
        if !test_line_interval_overapprox(x0, x1, λl, λu, n_test=n_test, tol=tol)
            push!(domains, (x0, x1, λl, λu))
        end
    end
    
    return domains
end


function test_bab_1d(init_domain, f, approx_f, maximize; max_steps=10, optimality_gap=1e-2, printing=false, n_test=100, tol=1e-12)
    xl, xu = init_domain

    xs = range(xl, xu, n_test)
    fs = f.(xs)

    fl = minimum(fs)
    fu = maximum(fs)

    eval_f = dom -> begin
        xl, xu = dom
        x = 0.5 * (xl + xu)
        f(x)
    end

    split_f = dom -> begin
        xl, xu = dom
        x_mid = 0.5 * (xl + xu)
        xl1 = xl
        xu1 = x_mid
        xl2 = x_mid
        xu2 = xu
        return (xl1, xu1), (xl2, xu2)
    end

    bnd = DP.generic_bab([init_domain], eval_f, approx_f, split_f, maximize, max_steps=max_steps, optimality_gap=optimality_gap, printing=printing)

    if maximize && fu > bnd + tol
        println("Bound must be larger than maximum! Got ", bnd, " < ", fu)
        return false
    elseif !maximize && fl < bnd - tol
        println("Bound must be smaller than minimum! Got ", bnd, " > ", fl)
        return false
    else
        return true
    end  
end


"""
params:
    make_f - function that takes (a,b,c) and returns function to evaluate overapproximation
    make_approx_f - function that takes (x0, x1, y0, y1, a, b, c) and returns overapproximating function for interval [λl, λu]
"""
function test_bab_line(make_f, make_approx_f, maximize; n_samples=100, max_steps=10, optimality_gap=1e-2, n_test=100, tol=1e-12)
    as = randn(n_samples)
    bs = randn(n_samples)
    cs = randn(n_samples)

    x0s = randn(n_samples)
    x1s = randn(n_samples)
    y0s = randn(n_samples)
    y1s = randn(n_samples)
    λls = rand(n_samples)
    λus = min.(1., λls .+ rand(n_samples))

    params = []
    for (a, b, c, x0, x1, y0, y1, λl, λu) in zip(as, bs, cs, x0s, x1s, y0s, y1s, λls, λus)
        init_dom = (λl, λu)

        f = make_f(a, b, c)

        f_line = λ -> begin
            x = (1 - λ)*x0 + λ*x1
            y = (1 - λ)*y0 + λ*y1
            f(x,y)
        end

        approx_f = make_approx_f(x0, x1, y0, y1, a, b, c)

        if !test_bab_1d(init_dom, f_line, approx_f, maximize, max_steps=max_steps, optimality_gap=optimality_gap, n_test=n_test, tol=tol)
            push!(params, (a, b, c, x0, x1, y0, y1, λl, λu))
        end
    end

    return params
end


function test_bab_line_maximize_σ_y(; n_samples=100, max_steps=10, optimality_gap=1e-2, n_test=100, tol=1e-12)
    make_f = (a,b,c) -> begin
        f(x,y) = L.σ(x)*y - (a*x + b*y + c)
    end

    make_approx_f = (x0, x1, y0, y1, a, b, c) -> begin
        approx_f = dom -> begin
            λl, λu = dom
            l, u = DP.interval_σy_lin(x0, x1, y0, y1, λl, λu, a, b, c)
            u
        end

        return approx_f
    end

    test_bab_line(make_f, make_approx_f, true, n_samples=n_samples, max_steps=max_steps, optimality_gap=optimality_gap, n_test=n_test, tol=tol)
end


function test_bab_line_minimize_σ_y(; n_samples=100, max_steps=10, optimality_gap=1e-2, n_test=100, tol=1e-12)
    make_f = (a,b,c) -> begin
        f(x,y) = L.σ(x)*y - (a*x + b*y + c)
    end

    make_approx_f = (x0, x1, y0, y1, a, b, c) -> begin
        approx_f = dom -> begin
            λl, λu = dom
            l, u = DP.interval_σy_lin(x0, x1, y0, y1, λl, λu, a, b, c)
            l
        end

        return approx_f
    end

    test_bab_line(make_f, make_approx_f, false, n_samples=n_samples, max_steps=max_steps, optimality_gap=optimality_gap, n_test=n_test, tol=tol)
end


function test_bab_line_maximize_σ_tanh(; n_samples=100, max_steps=10, optimality_gap=1e-2, n_test=100, tol=1e-12)
    make_f = (a,b,c) -> begin
        f(x,y) = L.σ(x)*tanh(y) - (a*x + b*y + c)
    end

    make_approx_f = (x0, x1, y0, y1, a, b, c) -> begin
        approx_f = dom -> begin
            λl, λu = dom
            l, u = DP.interval_σtanh_lin(x0, x1, y0, y1, λl, λu, a, b, c)
            u
        end

        return approx_f
    end

    test_bab_line(make_f, make_approx_f, true, n_samples=n_samples, max_steps=max_steps, optimality_gap=optimality_gap, n_test=n_test, tol=tol)
end


function test_bab_line_minimize_σ_tanh(; n_samples=100, max_steps=10, optimality_gap=1e-2, n_test=100, tol=1e-12)
    make_f = (a,b,c) -> begin
        f(x,y) = L.σ(x)*tanh(y) - (a*x + b*y + c)
    end

    make_approx_f = (x0, x1, y0, y1, a, b, c) -> begin
        approx_f = dom -> begin
            λl, λu = dom
            l, u = DP.interval_σtanh_lin(x0, x1, y0, y1, λl, λu, a, b, c)
            l
        end

        return approx_f
    end

    test_bab_line(make_f, make_approx_f, false, n_samples=n_samples, max_steps=max_steps, optimality_gap=optimality_gap, n_test=n_test, tol=tol)
end


function test_critical_points_zono(z::Zonotope, f, approx_f; n_test=1000)
    a, b, c, ϵ = approx_f(z, n_samples=100, max_steps=1000, optimality_gap=1e-6, printing=false)
    errfun(x,y) = f(x,y) - (a*x + b*y + c)

    xs = zeros(n_test)
    ys = zeros(n_test)
    errs = zeros(n_test)
    for i in 1:n_test
        g = 2 .* rand(ngens(z)) .- 1
        x, y = z.generators * g .+ z.center
        
        errs[i] = errfun(x,y)
        xs[i] = x
        ys[i] = y
    end

    ϵₗ_sample = minimum(errs)
    iₗ = argmin(errs)
    ϵᵤ_sample = maximum(errs)
    iᵤ = argmax(errs)

    if abs(ϵₗ_sample) > ϵ || abs(ϵᵤ_sample) > ϵ
        if abs(ϵₗ_sample) > ϵ
            println("Minimum must be contained in ϵ! Got $(ϵₗ_sample) and ϵ = $ϵ")
            println("\tx = $(xs[iₗ]), y = $(ys[iₗ])")
        end

        if abs(ϵᵤ_sample) > ϵ
            println("Maximum must be contained in ϵ! Got $(ϵᵤ_sample) and ϵ = $ϵ")
            println("\tx = $(xs[iᵤ]), y = $(ys[iᵤ])")
        end

        println("\tdomain = ", z)
        println("\tcoeffs = ", (a, b, c))
        return false
    end

    return true  
end


"""
Tests correctness of overapproximation of a function f(x,y) by a linear function approx_f(x,y) = a*x + b*y + c ± ϵ over zonotopic input domains.

Test is done by computing values of f(x,y) - (a*x + b*y + c) for valid samples in the zonotopic input domain and checking if they fall withing ϵ
from the linear approximation's values.
"""
function test_critical_points_zono(f, approx_f; n_samples=10000, n_test=10000)
    params = []
    for i in 1:n_samples
        n = rand(5:100)
        G = randn(2, n)
        c = randn(2)
        z = Zonotope(c, G)
        res = test_critical_points_zono(z, f, approx_f, n_test=n_test)
        if !res
            push!(params, z)
        end
    end

    return params
end


function test_critical_points_zono_σ_tanh(; n_samples=1000, n_test=10000)
    return test_critical_points_zono((x,y) -> L.σ(x)*tanh(y), DP.get_relaxation_σtanh_zono)
end


function test_critical_points_zono_σ_y(; n_samples=1000, n_test=10000)
    return test_critical_points_zono((x,y) -> L.σ(x)*y, DP.get_relaxation_σy_zono)
end


function test_critical_points_box(lx, ux, ly, uy, a, b, c, f, approx_f; n_test=1000)
    xs, ys = approx_f(lx, ux, ly, uy, a, b, c)
    errfun(x,y) = f(x,y) - (a*x + b*y + c)

    x = L.sample_uniform_bounds(lx, ux, n_test)
    y = L.sample_uniform_bounds(ly, uy, n_test)
    errs = errfun.(x, y)
    ϵₗ_sample = minimum(errs)
    iₗ = argmin(errs)
    ϵᵤ_sample = maximum(errs)
    iᵤ = argmax(errs)

    errs = errfun.(xs, ys)
    ϵₗ_crit = minimum(errs)
    ϵᵤ_crit = maximum(errs)

    if ϵₗ_sample < ϵₗ_crit || ϵᵤ_sample > ϵᵤ_crit
        if ϵₗ_sample < ϵₗ_crit
            println("true minimum must be smaller than sampled min! Got $(ϵₗ_sample) < $(ϵₗ_crit)")
            println("\tx = $(x[iₗ]), y = $(y[iₗ])")
        end

        if ϵᵤ_sample > ϵᵤ_crit
            println("true maximum must be larger than sample max! Got $(ϵᵤ_sample) > $(ϵᵤ_crit)")
            println("\tx = $(x[iᵤ]), y = $(y[iᵤ])")
        end

        println("\tdomain = ", [lx, ux], " × ", [ly, uy])
        println("\tcoeffs = ", (a, b, c))
        return false
    end

    return true  
end


function test_critical_points_box(f, approx_f; n_samples=10000, n_test=10000)
    params = []
    for i in 1:n_samples
        lx = randn()
        ux = lx + abs(randn())
        ly = randn()
        uy = ly + abs(randn())
        a = randn()
        b = randn()
        c = randn()
        res = test_critical_points_box(lx, ux, ly, uy, a, b, c, f, approx_f, n_test=n_test)
        if !res
            push!(params, (lx, ux, ly, uy, a, b, c))
        end
    end

    return params
end


function test_critical_points_σ_y(; n_samples=10000, n_test=10000)
    f(x,y) = L.σ(x)*y
    approx_f = L.get_critical_points_σ_y

    return test_critical_points_box(f, approx_f, n_samples=n_samples, n_test=n_test)
end


function test_critical_points_σ_tanh(; n_samples=10000, n_test=10000)
    f(x,y) = L.σ(x)*tanh(y)
    approx_f = L.get_critical_points_σ_tanh

    return test_critical_points_box(f, approx_f, n_samples=n_samples, n_test=n_test)
end


