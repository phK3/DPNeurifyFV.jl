# In order to simplify solver development, we want to convert as many theoretically linear/affine layers as possible to 
# actual dense layers, s.t. solver developers can concentrate on nonlinearities

"""
Convert a theoretically affine operation op into representation op(x) = Ax + b (for vectorized x).

Only operations that allow for batched inputs are supported!

args:
    op - a theoretically affine operation that supports batch input
    x - an example input for op in the right shape

kwargs:
    batched - true iff the op supports batched arguments

returns:
    A, b s.t. vec(op(x)) == A*vec(x) + b
"""
function affop2mat(op, x; batched=true)
    s_in = size(x)

    # want x with batch dim, but s_in without batch dim
    if s_in[end] != 1
        # append batch dimension to e.g. feed it through convolution
        x = reshape(x, s_in..., 1)
    else
        # don't want trailing 1, when we construct new shapes,
        # but directly want product-batch-dim
        s_in = s_in[1:end-1]
    end
    
    # get bias
    b = op(zero(x))

    s_out = size(b)

    if batched
        # identity matrix, but in shape of x with batch dim for every input
        eye = reshape(I(prod(s_in)), s_in..., prod(s_in))
        A = op(eye)
        A = reshape(A, prod(s_out), prod(s_in))
    else
        n_in = prod(s_in)
        A = zeros(prod(s_out), n_in)
        for i in 1:n_in
            eᵢ = (1:n_in) .== i  # i-th unit basis vector
            
            # build the matrix column by column
            A[:,i] .= op(eᵢ)
        end
    end

    b = reshape(b, prod(s_out))
    # for every entry, executing the layer also added the bias term to the result, but we only want the
    # linear contribution in the matrix
    A = A .- b  
    
    return A, b
end


"""
Convert a theoretically affine operation op(x₁, x₂, ..., xₙ) with multiple arguments into matrix representation A*[x₁; x₂; ... ; xₙ] + b.

Only operations that allow for batched input are supported!

args:
    op - theoretically affine operation supporting batched inputs
    args - example inputs for each input argument of op in the expected shapes

kwargs:
    batched - true iff op supports batched input

returns:
    A, b s.t. vec(op(x₁, x₂, ..., xₙ)) == A * vcat(vec(x₁), vec(x₂), ..., vec(xₙ)) + b
"""
function affop2mat(op, args...; batched=true)
    As = []
    bs = []
    for (i, arg) in enumerate(args)
        A, b = affop2mat(x -> op((j != i ? zero(args[j]) : x for j in 1:length(args))...), arg, batched=batched)
        push!(As, A)
        push!(bs, b)
    end

    A = hcat(As...)
    b = bs[end]
    return A, b    
end


"""
Converts theoretically affine single vector input node L to a Linear layer.

args:
    L - theoretically affine layer
    x - example input in the expected shape

kwargs:
    sparse_threshold - if nnz(A)/length(A) > sparse_threshold use a sparse matrix 
    double_precision - if true, use float64 for linear layer weights

returns:
    list of Linear layer L̂, s.t. vec(L(x)) = L̂(vec(x))
"""
function node2dense(L::Node, x; sparse_threshold=0.9, double_precision=false)
    A, b = affop2mat(x -> forward_node(ConcreteExecution(), L, x), x, batched=batched_nodes[typeof(L)])

    if sum(A .== 0)/length(A) > sparse_threshold
        A = sparse(A)
    end

    return [Linear(L.inputs, L.outputs, L.name, A, b, double_precision=double_precision)]
end


function node2dense(L::Node, x...; sparse_threshold=0.9, double_precision=false)
    A, b = affop2mat((x...) -> forward_node(ConcreteExecution(), L, x...), x..., batched=batched_nodes[typeof(L)])
   
    if sum(A .== 0)/length(A) > sparse_threshold
        A = sparse(A)
    end

    # assumes that input in converted net is already vectorized!!!
    concat_outs = [L.name * "_input_concat"]
    concat_layer = Concat(L.inputs, concat_outs, L.name * "_concat", 1)
    linear_layer = Linear(concat_outs, L.outputs, L.name, A, b, double_precision=double_precision)

    return [concat_layer, linear_layer]
end


function cg2dense(cg::CompGraph; sparse_threshold=0.9, double_precision=false, verbosity=0)
    in_shape = map(s -> ifelse(typeof(s) <: Integer, s, 1), cg.input_shape)
    x = zeros(in_shape)
    ydict = propagate(ConcreteExecution(), cg, x, return_dict=true)
    ydict["input"] = x  # why is this not stored?

    in_node = cg.in_node
    out_node = cg.out_node
    out_shape = cg.output_shape

    nodes = []
    for (key, node) in cg.nodes
        verbosity > 0 && println("Converting: ", key)
        xs = collect_inputs(cg, node, ydict)

        if linear_nodes[typeof(node)]
            converted_nodes = node2dense(node, xs..., sparse_threshold=sparse_threshold, double_precision=double_precision)

            if node == in_node
                in_node = first(converted_nodes)
            elseif node == out_node
                out_node = last(converted_nodes)
            end

            push!(nodes, converted_nodes...)
        else
            # don't convert nonlinear nodes
            push!(nodes, node)
        end
    end

    return CompGraph(nodes, in_node, out_node, prod(in_shape), prod(out_shape))
end