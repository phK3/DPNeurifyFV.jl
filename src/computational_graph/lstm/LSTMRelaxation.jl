
# TODO rename σ s.t. it doesn't collide with LazySets or implement as separate module

module LSTMRelaxation

using PolynomialRoots, LinearAlgebra

# sigmoid and its inverse
σ = x -> 1 / (1 + exp(-x))
σinv = y -> log(y / (1 - y))



function sample_uniform_bounds(l, u, n)
    return rand(n) .* (u - l) .+ l
end


"""
Calculates linear approximation of function f(x,y) by sampling points in box-bounds of x, y
and fitting least squares approximation.

Returns vector β s.t. β₁x + β₂y + β₃ is the linear approximation
"""
function get_linear_approximation(lx, ux, ly, uy, f; n_samples=100)
    xs = sample_uniform_bounds(lx, ux, n_samples)
    ys = sample_uniform_bounds(ly, uy, n_samples)
    
    X = [xs ys ones(n_samples)]
    y = f.(xs, ys)
    
    β = X \ y
    return β
end



"""
Returns the critical points σ(x)*tanh(y) - (a*x + b*y + c) over (x,y) ∈ [lx, ux] × [ly, uy].

args:
    lx - lower bound of x
    ux - upper bound of x
    ly - lower bound of y
    uy - upper bound of y
    a - x-coeff of linear function
    b - y-coeff of linear function
    c - bias of linear function

returns:
    xs - list of x-values of critical points
    ys - list of y-values of critical points
"""
function get_critical_points_σ_tanh(lx, ux, ly, uy, a, b, c)
    # candidates to be checked for maximum
    # start with the corners, we need to check them anyways
    xs = [lx, lx, ux, ux]
    ys = [ly, uy, ly, uy]

    # max at boundary x
    for x in [lx, ux]
        γ = b / σ(x)
        # solve 1 - t - γ = 0 ⇔ (1 - γ) - t = 0
        ts = roots([1 - γ, -1])
        t = Float64.(ts[1])  # should only have one solution
        if abs(t) <= 1
            y = atanh(t)
            if ly <= y && y <= uy
                push!(xs, x)
                push!(ys, y)
            end
        end
    end

    # max at boundary y
    for y in [ly, uy]
        γ = a / tanh(y)
        # solve s(1-s) = γ ⇔ -γ + s - s² = 0
        ss = roots([-γ, 1, -1])
        s = Float64.(filter(x -> imag(x) == 0, ss))  # can have up to 2 solutions
        # TODO: also handle for other solution
        for ŝ in s
            # ŝ is valid sigmoid value
            if ŝ >= 0 && ŝ <= 1
                x = σinv(ŝ)
                if lx <= x && x <= ux
                    push!(xs, x)
                    push!(ys, y)
                end
            end
        end
    end

    # max in interior
    ss = roots([-a^2, -b, 1+2*b, -b-2, 1])
    s = Float64.(filter(x -> imag(x) == 0, ss))
    t = a ./ (s .* (1 .- s))
    for (ŝ, t̂) in zip(s, t)
        if ŝ >= 0 && ŝ <= 1 && abs(t̂) <= 1
            x = σinv(ŝ)
            y = atanh(t̂)
            if lx <= x && x <= ux && ly <= y && y <= uy
                push!(xs, x)
                push!(ys, y)
            end
        end
    end
    
    return xs, ys
end


"""
Computes zonotope relaxation for σ(x)*tanh(y) over (x,y) ∈ [lx, ux] × [ly, uy].

The relaxation is a*x + b*y + c + ϵ, s.t. 
a*x + b*y + c - ϵ ≤ σ(x)*tanh(y) ≤ a*x + b*y + c + ϵ

returns:
    a - x-coefficient of the relaxation
    b - y-coefficient of the relaxation
    c - bias of the relaxation
    ϵ - error value of the relaxation
"""
function get_relaxation_σ_tanh(lx, ux, ly, uy; n_samples=100)
    h(x,y) = σ(x)*tanh(y)
    a, b, c = get_linear_approximation(lx, ux, ly, uy, h, n_samples=n_samples)
    xs, ys = get_critical_points_σ_tanh(lx, ux, ly, uy, a, b, c)
    ϵs = h.(xs, ys) .- (a .* xs .+ b .* ys .+ c)
    
    ϵₗ = minimum(ϵs)
    ϵᵤ = maximum(ϵs)

    c += 0.5 * (ϵₗ + ϵᵤ)
    ϵ = 0.5 * (ϵᵤ - ϵₗ)
    
    return a, b, c, ϵ
end


"""
Returns the critical points σ(x)*y - (a*x + b*y + c) over (x,y) ∈ [lx, ux] × [ly, uy].

args:
    lx - lower bound of x
    ux - upper bound of x
    ly - lower bound of y
    uy - upper bound of y
    a - x-coeff of linear function
    b - y-coeff of linear function
    c - bias of linear function

returns:
    xs - list of x-values of critical points
    ys - list of y-values of critical points
"""
function get_critical_points_σ_y(lx, ux, ly, uy, a, b, c)
    # candidates to be checked for maximum
    # start with the corners, we need to check them anyways
    xs = [lx, lx, ux, ux]
    ys = [ly, uy, ly, uy]

    # max at boundary y
    for y in [ly, uy]
        γ = a / y
        # solve s(1-s) = γ ⇔ -γ + s - s² = 0
        ss = roots([-γ, 1, -1])
        # can have up to 2 solutions, only want real solutions
        s = Float64.(filter(x -> imag(x) == 0, ss))
        # TODO: also handle for other solution
        for ŝ in s
            # ŝ is valid sigmoid value
            if ŝ >= 0 && ŝ <= 1
                x = σinv(ŝ)
                if lx <= x && x <= ux
                    push!(xs, x)
                    push!(ys, y)
                end
            end
        end
    end
    
    return xs, ys
end


"""
Computes zonotope relaxation for σ(x)*y over (x,y) ∈ [lx, ux] × [ly, uy].

The relaxation is a*x + b*y + c + ϵ, s.t. 
a*x + b*y + c - ϵ ≤ σ(x)*y ≤ a*x + b*y + c + ϵ

returns:
    a - x-coefficient of the relaxation
    b - y-coefficient of the relaxation
    c - bias of the relaxation
    ϵ - error value of the relaxation
"""
function get_relaxation_σ_y(lx, ux, ly, uy; n_samples=100)
    g(x,y) = σ(x)*y
    a, b, c = get_linear_approximation(lx, ux, ly, uy, g, n_samples=n_samples)
    xs, ys = get_critical_points_σ_y(lx, ux, ly, uy, a, b, c)
    ϵs = g.(xs, ys) .- (a .* xs .+ b .* ys .+ c)
    
    ϵₗ = minimum(ϵs)
    ϵᵤ = maximum(ϵs)

    c += 0.5 * (ϵₗ + ϵᵤ)
    ϵ = 0.5 * (ϵᵤ - ϵₗ)

    return a, b, c, ϵ
end


export get_relaxation_σ_tanh, get_relaxation_σ_y

end # module