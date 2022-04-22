
"""
interval_map(W::Matrix, l::AbstractVecOrMat, u::AbstractVecOrMat)

Simple linear mapping on intervals.
`L, U := ([W]₊*l + [W]₋*u), ([W]₊*u + [W]₋*l)`

Outputs:
- `(lbound, ubound)` (after the mapping)
"""
function NV.interval_map(W::AbstractMatrix{N}, l::AbstractVecOrMat, u::AbstractVecOrMat) where N
    W_pos = max.(W, zero(N))
    W_neg = min.(W, zero(N))
    l_new = W_pos * l + W_neg * u
    u_new = W_pos * u + W_neg * l

    # original code calculates max and min twice, which is unnecessary
    #l_new = max.(W, zero(N)) * l + min.(W, zero(N)) * u
    #u_new = max.(W, zero(N)) * u + min.(W, zero(N)) * l
    return (l_new, u_new)
end