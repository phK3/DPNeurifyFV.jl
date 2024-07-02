

function summarize_linear_layers(cg::CompGraph, n::Linear, output, summarized_names; verbosity=0, double_precision=false)
    n_out = length(n.dense.bias)
    summarize_linear_layers(cg, n, I(n_out), zeros(n_out), output, summarized_names, verbosity=verbosity, double_precision=double_precision)
end


function summarize_linear_layers(cg::CompGraph, n::Linear, W::AbstractArray, b::AbstractArray, output, summarized_names; verbosity=0, double_precision=false)
    verbosity > 1 && println("enter ", n.name, " (linear)")
    # can only have one input node
    # is of course linear
    Wₗ = n.dense.weight
    bₗ = n.dense.bias
    
    Ŵ = W*Wₗ
    b̂ = W*bₗ + b

    in_node = n.inputs[1] == "input" ? "input" : get_producer(cg, n.inputs[1])
    @show in_node
    @show cg.in_node

    # TODO: can't just check length(cg.in_node.outputs) as it is only 1 output, but it is used by many nodes!!!
    if in_node == cg.in_node && cg.usage_map[n.inputs[1]] > 1
            verbosity > 0 && println("case: Input node with > 1 outputs")
            # don't want to reduce input node, still want there to be only 1 input node
            # if we reduced, we would get one input node for each downstream node
            verbosity > 0 && println("\tsummarize: ", summarized_names)
            nodes = Vector{Node}()
            node = Linear(n.inputs, output, join(summarized_names, "+"), Ŵ, b̂, double_precision=double_precision)
            push!(nodes, node)
            return nodes
    elseif in_node == "input"
        # now we are at the input node and it had only one output
        # can't reduce further
        summarized_names = [summarized_names; n.name]
        verbosity > 0 && println("\tsummarize: ", summarized_names)
        node = Linear(["input"], output, join(summarized_names, "+"), Ŵ, b̂, double_precision=double_precision)
        return [node]
    else
        return summarize_linear_layers(cg, in_node, Ŵ, b̂, output, [summarized_names; n.name], verbosity=verbosity, double_precision=double_precision)
    end
end


function summarize_linear_layers(cg::CompGraph, n::Node, output, summarized_names; verbosity=0, double_precision=false)
    # only enter the version without W, b when there was no prior linear layer
    verbosity > 0 && println("enter ", n.name, " (non-linear)")
    nodes = Vector{Node}()  # have to give type, otherwise it will only have the type of n
    push!(nodes, n)
    for in_arg in n.inputs
        if in_arg != "input"
            # if in_arg == input, we can just do nothing 
            in_node = get_producer(cg, in_arg)
            summarized_nodes = summarize_linear_layers(cg, in_node, [in_arg], [], verbosity=verbosity, double_precision=double_precision)
            nodes = [nodes; summarized_nodes]
        end
    end

    return nodes
end


"""
Summarizes Linear layers in the parent-graph of the current node n.

args:
    cg - the computational graph which n is part of
    n - the current node (that is not a Linear layer)
    W - the summarized weight matrix connecting this node's output to output
    b - the summarized bias vector connecting this node's output to output
    summarized_names - vector of names of the already summarized layer between this node and the node whose input is output

kwargs:
    verbosity - print information
    double_precision - whether to use double precision for the summarized linear layers

returns:
    Summarized nodes contained in CompGraph from output to the parents of this node.
"""
function summarize_linear_layers(cg::CompGraph, n::Node, W::AbstractArray, b::AbstractArray, output, summarized_names; verbosity=0, double_precision=false)
    verbosity > 0 && println("enter ", n.name, " (non-linear)")
    verbosity > 0 && println("\tsummarize: ", summarized_names)
    node = Linear(n.outputs, output, join(summarized_names, "+"), W, b, double_precision=double_precision)
    
    nodes = Vector{Node}()
    push!(nodes, n)
    push!(nodes, node)
    for in_arg in n.inputs
        if in_arg != "input"
            # if in_arg == input, we can just do nothing 
            in_node = get_producer(cg, in_arg)
            summarized_nodes = summarize_linear_layers(cg, in_node, [in_arg], [], verbosity=verbosity, double_precision=double_precision)
            nodes = [nodes; summarized_nodes]
        end
    end

    return nodes
end


"""
Summarizes adjacent Linear layers into a single Linear layer in the given computational graph.

Note that only Linear layers (not layers that are theoretically linear or affine like Conv) are summarized!

If the first node has multiple outputs it is not summarized with its child nodes to maintain a single input node.
Nodes with multiple inputs are also not summarized, even if they are linear.
Non-linear nodes act as a barrier for summarization.

args:
    cg - the computational graph for which Linear layers are to be summarized

kwargs:
    verbosity - print message when entering each node and which nodes are summarized
    double_precision - whether to use double precision for the new layers

returns:
    CompGraph with summarized linear layers
"""
function summarize_linear_layers(cg::CompGraph; verbosity=0, double_precision=false)
    in_shape = cg.input_shape
    out_shape = cg.output_shape

    nodes = summarize_linear_layers(cg, cg.out_node, cg.out_node.outputs, [], verbosity=verbosity, double_precision=double_precision)

    if maximum([cg.usage_map[o] for o in cg.in_node.outputs]) > 1
        # we have to add it manually here, we skip adding it in the call above as it would be added by every of its child nodes
        # TODO: there has to be a better way!
        push!(nodes, cg.in_node)
    end

    in_nodes = filter(x -> "input" in x.inputs, collect(values(nodes)))
    out_nodes = filter(x -> x.outputs == cg.out_node.outputs, collect(values(nodes)))
    @assert length(in_nodes) == 1 "Only networks with a unique input node are supported! Got input nodes $(in_nodes)"
    @assert length(out_nodes) == 1 "Only networks with a unique output node are supported! Got output nodes $(out_nodes)"

    return CompGraph(nodes, in_nodes[1], out_nodes[1], in_shape, out_shape)
end