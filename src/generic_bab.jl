

"""
Generic branch and bound implementation for maximizing function.

args:
    init_domains - list of initial domains 
    eval_f - function mapping domain to a concrete output value of the function over that domain
    approx_f - function mapping domain to a valid upper bound of the function over that domain
    split_f - function splitting a domain into 2 parts

kwargs:
    max_steps - maximum number of steps 
    optimality_gap - distance between best concrete value and upper bound

returns:
    yu - certified upper bound on the function
    dom_star - domain containing best found concrete maximizer
"""
function generic_bab(init_domains, eval_f, approx_f, split_f; max_steps=10, optimality_gap=1e-2, printing=false)
    queue = PriorityQueue(Base.Order.Reverse)

    # domain with maximizing input
    dom_star = init_domains[1]
    yl = -Inf
    yu = Inf
    n_doms = 0
    for dom in init_domains
        y = eval_f(dom)

        if y > yl
            yl = y
            dom_star = dom
        end

        yu = approx_f(dom)
        # all keys must be distinct, so include time in the key
        #enqueue!(queue, dom, (yu, n_doms))
        push!(queue, dom => (yu, n_doms))
        n_doms += 1

        printing && println("enqueuing ", [y, yu])
    end

    for i in 1:max_steps
        dom, (yu, timestamp) = peek(queue)

        if yu - yl <= optimality_gap
            printing && println("Found optimal value ∈ ", [yl, yu])
            return yu, dom_star
        end
    
        dequeue!(queue)
    
        printing && println(i, ": max_λ ∈ ", [yl, yu])
    
        dom1, dom2 = split_f(dom)
        yu1 = approx_f(dom1)
        yu2 = approx_f(dom2)

        yl1 = eval_f(dom1)
        yl2 = eval_f(dom2)

        if (yl1 > yl ) && (yl1 >= yl2)
            dom_star = dom1
        elseif (yl2 > yl) && (yl2 > yl1)
            dom_star = dom2
        end

        yl = max(yl, yl1, yl2)
    
        yu1 > yl && enqueue!(queue, dom1, (yu1, n_doms + 1))
        yu2 > yl && enqueue!(queue, dom2, (yu2, n_doms + 2))
        n_doms += 2
    end
    
    return yu, dom_star
end


"""
Generic branch and bound implementation for optimizing a function.

args:
    init_domains - list of initial domains 
    eval_f - function mapping domain to a concrete output value of the function over that domain
    approx_f - if maximize is true, then function mapping domain to a valid upper bound of the function over that domain
               if maximize is false (minimization), then function mapping to a valid lower bound of the function over that domain
    split_f - function splitting a domain into 2 parts
    maximize - (bool) if true, then maximize the function, if false, minimize it.

kwargs:
    max_steps - maximum number of steps 
    optimality_gap - distance between best concrete value and upper bound

returns:
    yu - certified upper bound on the function
    dom_star - domain containing best found concrete maximizer
"""
function generic_bab(init_domains, eval_f, approx_f, split_f, maximize; max_steps=10, optimality_gap=1e-2, printing=false)
    eval_fun = eval_f
    approx_fun = approx_f
    if !maximize 
        eval_fun = dom -> -eval_f(dom)
        approx_fun = dom -> -approx_f(dom)
    end

    yu, dom_star = generic_bab(init_domains, eval_fun, approx_fun, split_f; max_steps=max_steps, optimality_gap=optimality_gap, printing=printing)
    
    return maximize ? (yu, dom_star) : (-yu, dom_star)
end