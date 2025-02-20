
"""
Interval overapproximation [l, u] of l ≤ (1 - λ)*x0 + λ*x1 ≤ u with λ ∈ [λl, λu] and constants x0, x1
"""
function line_interval_overapprox(x0, x1, λl, λu)
    # (1 - λ)*x0 + λ*x1 = (x0 - x1)*λ + x0
    dx1x0 = x1 - x0
    if dx1x0 >= 0
        l = dx1x0 * λl + x0
        u = dx1x0 * λu + x0
    else
        l = dx1x0 * λu + x0
        u = dx1x0 * λl + x0
    end

    return l, u    
end


"""
Calculates interval overapproximation for all values of σ(x)y with x and y lying on the line between (x₀,y₀) and (x₁,y₁).

More precisely, we get an interval [l, u] with l ≤ σ((1 - λ) x₀ + λ x₁) * ((1 - λ) y₀ + λ y₁) ≤ u for λ ∈ [λl, λu].
"""
function interval_σy(x₀, x₁, y₀, y₁, λl, λu)
    lx, ux = line_interval_overapprox(x₀, x₁, λl, λu)
    ly, uy = line_interval_overapprox(y₀, y₁, λl, λu)
    
    # Sigmoid is monotonic
    lσ = Flux.σ(lx) 
    uσ = Flux.σ(ux)

    lσy = min(lσ * ly, lσ * uy, uσ * ly, uσ * uy)
    uσy = max(lσ * ly, lσ * uy, uσ * ly, uσ * uy)

    return lσy, uσy
end


"""
Calculates interval overapproximation for all values of σ(x)*tanh(y) with x and y lying on the line between (x₀,y₀) and (x₁,y₁).

More precisely, we get an interval [l, u] with l ≤ σ((1 - λ) x₀ + λ x₁) * tanh((1 - λ) y₀ + λ y₁) ≤ u for λ ∈ [λl, λu].
"""
function interval_σtanh(x₀, x₁, y₀, y₁, λl, λu)
    lx, ux = line_interval_overapprox(x₀, x₁, λl, λu)
    ly, uy = line_interval_overapprox(y₀, y₁, λl, λu)

    lσ = Flux.σ(lx) 
    uσ = Flux.σ(ux)
    ltanh = tanh(ly)
    utanh = tanh(uy)

    lσy = min(lσ * ltanh, lσ * utanh, uσ * ltanh, uσ * utanh)
    uσy = max(lσ * ltanh, lσ * utanh, uσ * ltanh, uσ * utanh)
    
    return lσy, uσy
end


"""
Computes interval overapproximation [l, u] with l ≤ σ(x)y - f(x,y) ≤ u for (x,y) on the line between (x₀,y₀) and (x₁,y₁).

The linear function f(x,y) = a*x + b*y + c is usually an approximation to σ(x)y.
"""
function interval_σy_lin(x₀, x₁, y₀, y₁, λl, λu, a, b, c)
    lx, ux = line_interval_overapprox(x₀, x₁, λl, λu)
    ly, uy = line_interval_overapprox(y₀, y₁, λl, λu)

    fl, fu = interval_σy(x₀, x₁, y₀, y₁, λl, λu)
    lin_l = a*ifelse(a >= 0, lx, ux) + b*ifelse(b >= 0, ly, uy) + c
    lin_u = a*ifelse(a >= 0, ux, lx) + b*ifelse(b >= 0, uy, ly) + c

    outl = fl - lin_u
    outu = fu - lin_l

    return outl, outu
end


"""
Computes interval overapproximation [l, u] with l ≤ σ(x)*tanh(y) - f(x,y) ≤ u for (x,y) on the line between (x₀,y₀) and (x₁,y₁).

The linear function f(x,y) = a*x + b*y + c is usually an approximation to σ(x)*tanh(y).
"""
function interval_σtanh_lin(x₀, x₁, y₀, y₁, λl, λu, a, b, c)
    lx, ux = line_interval_overapprox(x₀, x₁, λl, λu)
    ly, uy = line_interval_overapprox(y₀, y₁, λl, λu)

    fl, fu = interval_σtanh(x₀, x₁, y₀, y₁, λl, λu)
    lin_l = a*ifelse(a >= 0, lx, ux) + b*ifelse(b >= 0, ly, uy) + c
    lin_u = a*ifelse(a >= 0, ux, lx) + b*ifelse(b >= 0, uy, ly) + c

    outl = fl - lin_u
    outu = fu - lin_l

    return outl, outu
end


"""
Construct linear segments along the boundary of a 2d zonotope.

A single segment with x = (1 - λ) x₀ + λ x₁, y = (1 - λ) y₀ + λ y₁ for λ ∈ [λl, λu] is denoted by (λl, λu, x₀, x₁, y₀, y₁).
All λl = 0 and λu = 1 to cover the whole line segment.

args:
    z - the zonotope to compute the linear segments for

returns:
    list of [(λl, λu, x₀, x₁, y₀, y₁)]
"""
function construct_initial_domains(z::Zonotope)
    @assert dim(z) == 2 "Can only construct domains for 2d zonotopes, found dim(z) == $(dim(z))"

    # need to remove redundant generators, otherwise we get duplicate line segments!
    z = remove_redundant_generators(z)
    z = reduce_order(z, 5)
    v = vertices_list_2d_zonotope(z)
    n_vertices = size(v, 2) - 1
    x0s = zeros(n_vertices)
    x1s = zeros(n_vertices)
    y0s = zeros(n_vertices)
    y1s = zeros(n_vertices)

    for i in 1:size(v, 2) - 1
        x0s[i] = v[1,i]
        x1s[i] = v[1,i+1]
        y0s[i] = v[2,i]
        y1s[i] = v[2,i+1]
    end

    initial_domains = [(0., 1., x₀, x₁, y₀, y₁) for (x₀, x₁, y₀, y₁) in zip(x0s, x1s, y0s, y1s)]
end


"""
Calculate linear overapproximation of σ(x)y over a 2d zonotopic input domain.

Samples the zonotope vertices and computes a linear approximation f(x,y) fitting the function values at those vertices.
Since σ(x)y - f(x) for linear f has no extrema in the interior, we just overapproximate the error over the boundary of the zonotope
and shift up or down by that value.

args:
    z - 2d zonotope describing the input domain

kwargs:
    n_samples - not needed
    max_steps - maximum number of steps in interval overapproximation on zonotope boundary
    optimality_gap - optimality gap for interval overapproximation
    printing - whether to print interval overapproximation progress

returns:
    a, b, c, ϵ - for linear overapproximation f(x,y) = a*x + b*y + c ± ϵ
"""
function get_relaxation_σy_zono(z; n_samples=100, max_steps=1000, optimality_gap=1e-6, printing=false, method=:remezlike)
    # no extrema in interior for σ(x)y
    #v = vertices_list_2d_zonotope(z)

    #xs = v[1,1:end-1]
    #ys = v[2,1:end-1]
    #X = [xs ys ones(size(v, 2)-1)]
    #y = Flux.σ.(xs) .* ys

    #a, b, c = LSTMRelaxation.linear_approximation_lp(X, y)

    l = low(z)
    u = high(z)
    a, b, c = LSTMRelaxation.linear_approximation_remezlike(l[1], u[1], l[2], u[2], (x,y) -> Flux.σ(x)*y, :σy)

    initial_domains = construct_initial_domains(z)

    # define functions here because we need fixed a, b, c
    eval_f = dom -> begin
        λl, λu, x₀, x₁, y₀, y₁ = dom
        λ = 0.5 * (λl + λu)
        x = (1 - λ)*x₀ + λ*x₁
        y = (1 - λ)*y₀ + λ*y₁
        Flux.σ(x) * y - (a*x + b*y + c)
    end

    approx_f_max = dom -> begin
        λl, λu, x₀, x₁, y₀, y₁ = dom
        l, u = interval_σy_lin(x₀, x₁, y₀, y₁, λl, λu, a, b, c)
        u
    end

    # need underapprox for minimization
    approx_f_min = dom -> begin
        λl, λu, x₀, x₁, y₀, y₁ = dom
        l, u = interval_σy_lin(x₀, x₁, y₀, y₁, λl, λu, a, b, c)
        l
    end

    split_f = dom -> begin
        λl, λu, x₀, x₁, y₀, y₁ = dom
        λ_mid = 0.5 * (λl + λu)
        λl1 = λl
        λu1 = λ_mid
        λl2 = λ_mid
        λu2 = λu
        return (λl1, λu1, x₀, x₁, y₀, y₁), (λl2, λu2, x₀, x₁, y₀, y₁)
    end

    ϵₗ = generic_bab(initial_domains, eval_f, approx_f_min, split_f, false, max_steps=max_steps, optimality_gap=optimality_gap, printing=printing)
    ϵᵤ = generic_bab(initial_domains, eval_f, approx_f_max, split_f, max_steps=max_steps, optimality_gap=optimality_gap, printing=printing)

    c += 0.5 * (ϵₗ + ϵᵤ)
    ϵ = 0.5 * (ϵᵤ - ϵₗ)

    return a, b, c, ϵ
end


"""
Calculate linear overapproximation of σ(x)tanh(y) over a 2d zonotopic input domain.

Samples the zonotope vertices and computes a linear approximation f(x,y) fitting the function values at those vertices.
σ(x)tanh(y) - f(x) can have extrema in the interior of z!
Therefore, we solve for the extrema and test for their inclusion in z.
Additionally, we overapproximate the error over the boundary of the zonotope.
We shift up or down by the maximum approximation error.

args:
    z - 2d zonotope describing the input domain

kwargs:
    n_samples - not needed
    max_steps - maximum number of steps in interval overapproximation on zonotope boundary
    optimality_gap - optimality gap for interval overapproximation
    printing - whether to print interval overapproximation progress

returns:
    a, b, c, ϵ - for linear overapproximation f(x,y) = a*x + b*y + c ± ϵ
"""
function get_relaxation_σtanh_zono(z; n_samples=100, max_steps=1000, optimality_gap=1e-6, printing=false, method=:remezlike)
    # extrema in interior for σ(x)*tanh(y) are possible!!!
    #v = vertices_list_2d_zonotope(z)

    #xs = v[1,1:end-1]
    #ys = v[2,1:end-1]
    #X = [xs ys ones(size(v, 2)-1)]
    #y = Flux.σ.(xs) .* tanh.(ys)

    l = low(z)
    u = high(z)
    a, b, c = LSTMRelaxation.linear_approximation_remezlike(l[1], u[1], l[2], u[2], (x,y) -> Flux.σ(x)*tanh(y), :σtanh)
    #a, b, c = LSTMRelaxation.linear_approximation_lp(X, y)

    # extrema in interior
    interior_xs = Float64[]
    interior_ys = Float64[]
    ss = roots([-a^2, -b, 1+2*b, -b-2, 1])
    s = Float64.(filter(x -> imag(x) == 0, ss))
    t = a ./ (s .* (1 .- s))
    for (ŝ, t̂) in zip(s, t)
        if ŝ >= 0 && ŝ <= 1 && abs(t̂) <= 1
            x = LSTMRelaxation.σinv(ŝ)
            y = atanh(t̂)
            if [x,y] ∈ z 
                # TODO: check if this ∈ is efficient!!!
                push!(interior_xs, x)
                push!(interior_ys, y)
            end
        end
    end

    ϵs = Flux.σ.(interior_xs) .* tanh.(interior_ys) .- (a .* interior_xs .+ b .* interior_ys .+ c)
    ϵₗ_interior = minimum(ϵs, init=Inf)  # set ϵₗ=Inf if empty, s.t. it is easily overwritten later 
    ϵᵤ_interior = maximum(ϵs, init=-Inf)


    initial_domains = construct_initial_domains(z)

    # define functions here because we need fixed a, b, c
    eval_f = dom -> begin
        λl, λu, x₀, x₁, y₀, y₁ = dom
        λ = 0.5 * (λl + λu)
        x = (1 - λ)*x₀ + λ*x₁
        y = (1 - λ)*y₀ + λ*y₁
        Flux.σ(x) * tanh(y) - (a*x + b*y + c)
    end

    approx_f_max = dom -> begin
        λl, λu, x₀, x₁, y₀, y₁ = dom
        l, u = interval_σtanh_lin(x₀, x₁, y₀, y₁, λl, λu, a, b, c)
        u
    end

    approx_f_min = dom -> begin
        λl, λu, x₀, x₁, y₀, y₁ = dom
        l, u = interval_σtanh_lin(x₀, x₁, y₀, y₁, λl, λu, a, b, c)
        l
    end

    split_f = dom -> begin
        λl, λu, x₀, x₁, y₀, y₁ = dom
        λ_mid = 0.5 * (λl + λu)
        λl1 = λl
        λu1 = λ_mid
        λl2 = λ_mid
        λu2 = λu
        return (λl1, λu1, x₀, x₁, y₀, y₁), (λl2, λu2, x₀, x₁, y₀, y₁)
    end

    ϵₗ = generic_bab(initial_domains, eval_f, approx_f_min, split_f, false, max_steps=max_steps, optimality_gap=optimality_gap, printing=printing)
    ϵᵤ = generic_bab(initial_domains, eval_f, approx_f_max, split_f, max_steps=max_steps, optimality_gap=optimality_gap, printing=printing)

    ϵₗ = min(ϵₗ, ϵₗ_interior)
    ϵᵤ = max(ϵᵤ, ϵᵤ_interior)

    c += 0.5 * (ϵₗ + ϵᵤ)
    ϵ = 0.5 * (ϵᵤ - ϵₗ)

    return a, b, c, ϵ
end
