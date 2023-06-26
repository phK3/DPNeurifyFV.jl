

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
end


"""
Dummy solver for execution with concrete values.

(since forward_node() requires a solver argument)
"""
struct ConcreteExecution <: Solver end


function CompGraph(nodes::AbstractVector, in_node::Node, out_node::Node)
    @assert length(in_node.parents) == 0 "Input node should not have parents! Has $(in_node.parents)"
    @assert length(out_node.children) == 0 "Output node should not have children! Has $(out_node.children)"
    
    node_dict = Dict()
    for n in nodes
        if haskey(node_dict, n.name)
            throw(ArgumentError("name $(n.name) duplicate!"))
        end
        
        node_dict[n.name] = n
    end
    
    return CompGraph(node_dict, in_node, out_node)
end


# define getter functions to ensure that each Node has these features
# nodes only have list with identifiers
get_children(n::Node) = n.children
get_parents(n::Node) = n.parents
get_name(n::Node) = n.name

# real Node objects are stored in network dict
get_children(nn::CompGraph, n::Node) = [nn.nodes[cname] for cname in n.children]
get_parents(nn::CompGraph, n::Node) = [nn.nodes[pname] for pname in n.parents]


"""
Get inputs to the node in a network from the propagation dictionary.

We need to know the network, since it connects the names of the nodes with the actual nodes.
"""
function collect_inputs(nn::CompGraph, node::Node, prop_dict::Dict{K,V}) where {K, V}
    inputs = V[]
    for p in get_parents(nn, node)
        input = prop_dict[p.name]
        push!(inputs, input)
    end
    
    return inputs
end


"""
Propagates values given in the propagation dictionary through the computational graph.
The dictionary is modified in the process and contains the results of the computation.
"""
function propagate!(solver, nn::CompGraph, node::Node; prop_dict=nothing)
    # should we only allow non-empty or non-nothing dicts?
    prop_dict = isnothing(prop_dict) ? Dict() : prop_dict

    # TODO: only store intermediate results as long as they are needed (decrement counter of node's children?)
    # TODO: make intermediate results available to caller
    for p in get_parents(nn, node)
        if ~haskey(prop_dict, p.name)
            propagate!(solver, nn, p, prop_dict=prop_dict)
        end
    end
    
    inputs = collect_inputs(nn, node, prop_dict)
    ŝ = forward_node(solver, node, inputs...)
    
    # TODO: what about nodes with multiple outputs like Split?
    # put "Split_output_0", "Split_output_1" as separate entries into the dict
    prop_dict[node.name] = ŝ
end  


"""
Propagates an input through a computational graph and returns the result at the single output node.
"""
function propagate(solver, nn::CompGraph, x; return_dict=false)
    x̂ = forward_node(solver, nn.in_node, x)
    prop_dict = Dict(get_name(nn.in_node) => x̂)
    
    propagate!(solver, nn, nn.out_node, prop_dict=prop_dict)
    
    if return_dict
        return prop_dict
    else
        y = prop_dict[get_name(nn.out_node)]
        return y
    end
end


function propagate(nn::CompGraph, x)
    solver = ConcreteExecution()
    return propagate(solver, nn, x)
end
