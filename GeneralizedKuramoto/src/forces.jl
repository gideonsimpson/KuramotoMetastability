"""
`tabulate_nonlocal_force!`: Tabulate the nonlocal force term:

∑K(i,j)S(u(i,t),u(j,t))

with Euler time stepping:

### Fields
* `nonlocal_force` - Vector of values to be populated
* `K` - Connectivity matrix with weights, either dense or CSC format
* `S` - S = S(u(x), u(y)), interaction term
* `u` - Current state of the system
"""
function tabulate_nonlocal_force!(nonlocal_force, K::TK, S::TS, u) where {TK<:Matrix,TS}

    @. nonlocal_force = 0
    n = length(u)

    for i in 1:n, j in 1:n
        nonlocal_force[i] += K[i, j] * S(u[i], u[j])
    end

    nonlocal_force

end

function tabulate_nonlocal_force!(nonlocal_force, K::TK, S::TS, u) where {TK<:SparseMatrixCSC,TS}

    @. nonlocal_force = 0
    n = length(u)

    for j in 1:n
        for k in K.colptr[j]:K.colptr[j+1]-1
            i = K.rowval[k]
            nonlocal_force[i] += K.nzval[k] * S(u[i], u[j])
        end
    end

    nonlocal_force

end
