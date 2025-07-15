module ClassicalKuramoto

using SparseArrays

"""
`energy`: Compute the energy for the Kuramoto model,

E = ∑ (-1)^(1+α)/(4π) *cos(2π(uⱼ-uᵢ))

where the sum is taken over the neighbor set of site i

### Fields
* `u` - State of the system
* `k` - Number of neighbors to the left/right, for a total of 2k+1
* `α` - Sets attractive or repulsive coupling
"""
function energy(u, k::Integer, α)

    E = 0.0;

    n = length(u);

    for i in 1:n
        for j in mod1.(i-k:i+k,n)
            E += 0.5 * (-1)^(1+α)/(2*π) * cospi(2*(u[j]-u[i]))
        end
    end

    return E
end

"""
`energy`: Compute the energy for the Kuramoto model,

E = ∑ (-1)^(1+α)/(4π) *cos(2π(uⱼ-uᵢ))

where the sum is taken over the neighbor set of site i

### Fields
* `u` - State of the system
* `K` - Neighbors adjacency matrix
* `α` - Sets attractive or repulsive coupling
"""
function energy(u, K::TK, α) where{TK<:Matrix}

    E = 0.0;

    n = length(u);

    for i in 1:n, j in 1:n
        E += 0.5 * (-1)^(1+α)/(2*π) *K[i,j] *  cospi(2*(u[j]-u[i]))
    end

    return E
end

"""
`energy`: Compute the energy for the Kuramoto model,

E = ∑ (-1)^(1+α)/(4π) *cos(2π(uⱼ-uᵢ))

where the sum is taken over the neighbor set of site i

### Fields
* `u` - State of the system
* `K` - Neighbors adjacency matrix
* `α` - Sets attractive or repulsive coupling
"""
function energy(u, K::TK, α) where{TK<:SparseMatrixCSC}

    E = 0.0;

    n = length(u);

    for i in 1:n
        for k in K.colptr[i]:K.colptr[i+1]-1
            j = K.rowval[k];
            E += 0.5 * (-1)^(1+α)/(2*π) * K.nzval[k] *  cospi(2*(u[j]-u[i]))
        end
    end

    return E
end


"""
`grad_energy!`: Compute the energy gradient

(-1)^(1+α) ∑K(i,j)sin(2π(uⱼ-uᵢ))


### Fields
* `gradE` - Vector of values to be populated
* `u` - State of the system
* `k` - Number of neighbors to the left/right, for a total of 2k+1
* `α` - Sets attractive or repulsive coupling
### Optional Fields
* `scale = false` - Normalize by the number of neighbors, 2k+1
"""
function grad_energy!(gradE, u, k::Integer, α; scale=false)

    @. gradE = 0;
    n = length(u);
    C = 1.0;
    if scale
        C /= (2*k+1);
    end
    
    for i in 1:n
        for j in mod1.(i-k:i+k,n)
            gradE[i] += (-1)^(1+α)* C * sinpi(2*(u[j] - u[i]));
        end
    end

    gradE

end

"""
`grad_energy!`: Compute the energy gradient

(-1)^(1+α) ∑K(i,j)sin(2π(uⱼ-uᵢ))


### Fields
* `gradE` - Vector of values to be populated
* `u` - State of the system
* `K` - Neighbors adjacency matrix
* `α` - Sets attractive or repulsive coupling
"""
function grad_energy!(gradE, u, K::TK, α) where {TK<:Matrix}

    @. gradE = 0;
    n = length(u);
    
    for i in 1:n
        for j in 1:n
            gradE[i] += (-1)^(1+α) * K[i,j] * sinpi(2*(u[j] - u[i]));
        end
    end

    gradE

end



"""
`grad_energy!`: Compute the energy gradient

(-1)^(1+α) ∑K(i,j)sin(2π(uⱼ-uᵢ))


### Fields
* `gradE` - Vector of values to be populated
* `u` - State of the system
* `K` - Neighbors adjacency matrix
* `α` - Sets attractive or repulsive coupling
"""
function grad_energy!(gradE, u, K::TK, α) where {TK<:SparseMatrixCSC}

    @. gradE = 0;
    n = length(u);
    
    for i in 1:n
        for k in K.colptr[i]:K.colptr[i+1]-1
            j = K.rowval[k];
            gradE[i] += (-1)^(1+α) *  K.nzval[k] * sinpi(2*(u[j] - u[i]));
        end
    end

    gradE

end


end #end module