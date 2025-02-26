
using DPNeurifyFV, LazySets, NeuralVerification
import VNNLib.NNLoader as NNL

const NV = NeuralVerification
const DP = DPNeurifyFV

onnx_path = joinpath(@__DIR__, "../../MaxpoolVerify.jl/networks/net6x50_overfit.onnx")
nn = NNL.load_network_dict(DP.CGType, onnx_path)
nn = DP.cg2dense(nn)
nn = DP.summarize_linear_layers(nn)

y = DP.propagate(nn, zeros(nn.input_shape))

lbs = zeros(12)
ubs = ones(12)
ubs[1] = 0.  # smallest lb is fixed to zero
input_set = Hyperrectangle(low=lbs, high=ubs)

solver = DP.DPNFV(method=:DeepPolyRelax)
s = DP.init_symbolic_interval_graph(nn, input_set, max_vars=20)
ŝ = DP.propagate(solver, nn, s)

params = DP.PriorityOptimizerParameters(max_steps=100000, print_frequency=1000, stop_frequency=1, verbosity=2)
DP.optimize_linear(nn, input_set, [1.], params, solver=solver)


function check_validity_maxpool(s::DP.SymbolicIntervalGraph{<:Hyperrectangle}; h=2, w=2)
    lx = low(DP.domain(s))
    ux = high(DP.domain(s))

    ll = lx[1:h*w]
    lu = ux[1:h*w]
    ul = lx[h*w+1:2*h*w]
    uu = ux[h*w+1:2*h*w]
    al = lx[2*h*w+1:end]
    au = ux[2*h*w+1:end]

    # sortedness of lower bounds is violated
    any(ll[1:end-1] .> lu[2:end]) && return false

    # maximum of upper bounds == 1 is violated
    all(uu .< 1) && return false

    # there is an upper bound that is larger than the corresponding lower bound 
    any(ul .> lu) && return false

    return true
end

l = [1., 2, 4, 3, 0,0,0,0,0,0,0,0]
u = [2., 4, 5, 4, 1,1,1,1,1,1,1,1]
H = Hyperrectangle(low=l, high=u);


function tighten_input_validity(H::Hyperrectangle; h=2, w=2)
    lx = low(H)
    ux = high(H)

    ll = lx[1:h*w]
    lu = ux[1:h*w]

    # since we require sortedness by lower bound, we can tighten using the fact that 
    # lᵢ ≤ lᵢ₊₁ ==> set lower bound for lᵢ₊₁ to max(llᵢ, llᵢ₊₁)
    #           ==> set upper bound for lᵢ to min(luᵢ, luᵢ₊₁)
    l̂ = [accumulate(max, ll); lx[h*w+1:end]]
    û = [reverse(accumulate(min, reverse(lu))); ux[h*w+1:end]]

    #if (ll != l̂[1:4]) || (lu != û[1:4])
    #    @show ll
    #    @show lu
    #    @show l̂[1:4]
    #    @show û[1:4]
    #end

    return Hyperrectangle(low=l̂, high=û)
end

function split_valid_interval(s::DP.SymbolicIntervalGraph{<:Hyperrectangle})
    radius = high(DP.domain(s)) - low(DP.domain(s))
    most_important_dim = sum(s.importance) == 0. ? argmax(radius) : argmax(s.importance .* radius)
    domain1, domain2 = split(DP.domain(s), most_important_dim)

    domain1 = tighten_input_validity(domain1)
    domain2 = tighten_input_validity(domain2)

    current_n_vars = DP.get_n_vars(s)
    s1 = DP.init_symbolic_interval_graph(s, domain1; max_vars=current_n_vars)
    if domain1 == domain2
        return [s for s in [s1] if check_validity_maxpool(s)]
    else
        s2 = DP.init_symbolic_interval_graph(s, domain2; max_vars=current_n_vars)
        return [s for s in [s1, s2] if check_validity_maxpool(s)]
    end
end


function concrete_input_valid(s::DP.SymbolicIntervalGraph{<:Hyperrectangle}; h=2, w=2)
    lx = low(DP.domain(s))
    ux = high(DP.domain(s))
    width = ux .- lx

    l = lx[1:h*w] .+ width[1:h*w] .* rand(h*w)
    l = accumulate(max, l)

    u = lx[h*w+1:2*h*w] .+ width[h*w+1:2*h*w] .* rand(h*w)
    u = max.(l, u)
    i = argmax(u)
    u[i] = 1.

    a = lx[2*h*w+1:end] .+ width[2*h*w+1:end] .* rand(h*w)

    return [l; u; a]    
end


params = DP.PriorityOptimizerParameters(max_steps=1000000, print_frequency=500, stop_frequency=1, verbosity=2, timeout=300)
DP.optimize_linear(nn, input_set, [1.], params, solver=solver, split=split_valid_interval, concrete_sample=:valid)
