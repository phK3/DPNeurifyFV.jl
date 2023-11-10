

"""
    A node is guaranteed to include 
    - a list of identifiers for inputs (parents) 
    - a list of identifiers for outputs (children)
    - an identifier (name)
    - the params of the node
"""
abstract type Node end

struct CompGraph
    # names of nodes as keys
    nodes::Dict
    in_node::Node
    out_node::Node
    # dict: output names -> node producing that output
    out_dict::Dict
    input_shape
    output_shape
end


"""
Dummy solver for execution with concrete values.

(since forward_node() requires a solver argument)
"""
struct ConcreteExecution <: Solver end


function CompGraph(nodes::AbstractVector, in_node::Node, out_node::Node, input_shape, output_shape)
    @assert length(in_node.inputs) == 1 "Input node should not have the iput value as input! Has $(in_node.inputs)"
    @assert length(out_node.outputs) == 1 "Output node should only have one output value! Has $(out_node.outputs)"
    
    node_dict = Dict()
    output_dict = Dict()
    for n in nodes
        if haskey(node_dict, n.name)
            throw(ArgumentError("name $(n.name) duplicate!"))
        end
        
        node_dict[n.name] = n
        
        for o in n.outputs
            output_dict[o] = n
        end
    end

    parents_dict = Dict()
    for n in nodes
        if n.name == in_node.name
            # input node has no parent nodes, just the input values
            continue
        end
        # all nodes producing inputs that node n needs are parents of n
        parents_dict[n.name] = [output_dict[i] for i in n.inputs]
    end
        
    return CompGraph(node_dict, in_node, out_node, output_dict, input_shape, output_shape)
end


# define getter functions to ensure that each Node has these features
# nodes only have list with identifiers
get_outputs(n::Node) = n.outputs
get_inputs(n::Node) = n.inputs
get_name(n::Node) = n.name

# real Node objects are stored in network dict
# doesn't work like this anymore, we now store outputs not children of nodes
# get_children(nn::CompGraph, n::Node) = [nn.nodes[cname] for cname in n.children]
# get_parents(nn::CompGraph, n::Node) = nn.parents[n]
get_parents(nn::CompGraph, n::Node) = [nn.out_dict[i] for i in get_inputs(n)]

"""
Get node producing output o in network nn.
"""
get_producer(nn::CompGraph, o) = nn.out_dict[o]

get_input_shape(nn::CompGraph) = nn.input_shape
get_output_shape(nn::CompGraph) = nn.output_shape


"""
Get inputs to the node in a network from the propagation dictionary.

We need to know the network, since it connects the names of the nodes with the actual nodes.
"""
function collect_inputs(nn::CompGraph, node::Node, prop_dict::Dict{K,V}) where {K, V}
    #inputs = V[]
    #for i in get_inputs(node)
    #    input = prop_dict[i]
    #    push!(inputs, input)
    #end
    #
    #return inputs
    return [prop_dict[i] for i in get_inputs(node)]
end


function add_outputs!(prop_dict, outputs, out_names::AbstractVector)
    if length(out_names) == 1
        prop_dict[out_names[1]] = outputs
    else
        for (o, n) in zip(outputs, out_names)
            prop_dict[n] = o
        end
    end
end


"""
Propagates values given in the propagation dictionary through the computational graph.
The dictionary is modified in the process and contains the results of the computation.
"""
function propagate!(solver, nn::CompGraph, node::Node; prop_dict=nothing, verbosity=0)
    # should we only allow non-empty or non-nothing dicts?
    prop_dict = isnothing(prop_dict) ? Dict() : prop_dict

    # TODO: only store intermediate results as long as they are needed (decrement counter of node's children?)
    # TODO: make intermediate results available to caller
    for i in get_inputs(node)
        if ~haskey(prop_dict, i)
            p = get_producer(nn, i)
            propagate!(solver, nn, p, prop_dict=prop_dict, verbosity=verbosity)
        end
    end

    #for p in get_parents(nn, node)
    #    if ~haskey(prop_dict, p.name)
    #        propagate!(solver, nn, p, prop_dict=prop_dict)
    #    end
    #end
    
    inputs = collect_inputs(nn, node, prop_dict)
    verbosity > 0 && println("propagating ", node.name)
    verbosity > 1 && println("\t shape: ", [size(x) for x in inputs])
    ŝ = forward_node(solver, node, inputs...)

    
    # TODO: what about nodes with multiple outputs like Split?
    # put "Split_output_0", "Split_output_1" as separate entries into the dict
    add_outputs!(prop_dict, ŝ, get_outputs(node))
end  


"""
Propagates an input through a computational graph and returns the result at the single output node.
"""
function propagate(solver, nn::CompGraph, x; return_dict=false, verbosity=0)
    # have to initialize an empty dict, if we initialize it with x̂, then it has exactly that type
    prop_dict = Dict()
    x̂ = forward_node(solver, nn.in_node, x)
    add_outputs!(prop_dict, x̂, get_outputs(nn.in_node))
    # prop_dict[get_name(nn.in_node)] = x̂
    # prop_dict = Dict(get_name(nn.in_node) => x̂)
    
    propagate!(solver, nn, nn.out_node, prop_dict=prop_dict, verbosity=verbosity)
    
    if return_dict
        return prop_dict
    else
        # we only support one output!
        y = prop_dict[get_outputs(nn.out_node)[1]]
        return y
    end
end


function propagate(nn::CompGraph, x; verbosity=0)
    solver = ConcreteExecution()
    return propagate(solver, nn, x, verbosity=verbosity)
end
