module GraphMatrices

using SparseArrays
using QuadGK

"""
`discrete_knn_matrix`: Construct the nearest neigbhors matrix with 1's in
entries where
|i-j| ≤ k
and zeros elsewhere.  Returns a dense matrix.

### Fields
* `n` - Matrix is n×n
* `k` - 2⋅k+1 neighbors
"""
function discrete_knn_matrix(n, k; periodic=true)
    K = zeros(n,n);

    for i in 1:n, j in 1:n
        if(periodic)
            dist = min(abs(i-j), n - abs(i-j));
        else
            dist = abs(i-j);
        end

        if dist ≤ k
            K[i,j] = 1;
        end
    end
    return K
end

"""
`discrete_knn_spmatrix`: Construct the nearest neigbhors matrix with 1's in
entries where
|i-j| < n ⋅ r
and zeros elsewhere.  Returns a sparse CSC matrix.

### Fields
* `n` - Matrix is n×n
* `k` - 2⋅k+1 neighbors
"""
function discrete_knn_spmatrix(n, k; periodic=true)
    Is = Int[];
    Js = Int[];
    Vs = Float64[];

    for i in 1:n, j in 1:n
        if(periodic)
            dist = min(abs(i-j), n - abs(i-j));
        else
            dist = abs(i-j);
        end

        if dist ≤ k
            push!(Is,i);
            push!(Js,j);
            push!(Vs,1);
        end
    end
    return sparse(Is, Js, Vs)
end

"""
`continuous_knn_spmatrix`: Construct the nearest neigbhors matrix in the
continuous case with entries

∬1_|i-j + s-t|<nr dsdt

Returns a sparse CSC matrix.

### Fields
* `n` - Matrix is n×n
* `r` - 0<r<1 determines number of neighbors,

"""
function continuous_knn_spmatrix(n, r; periodic=true)
    Is = Int[];
    Js = Int[];
    Vs = Float64[];

    for i in 1:n, j in 1:n
        if(periodic)
            dist = min(abs(i-j), n - abs(i-j));
        else
            dist = abs(i-j);
        end

        if dist <= n*r -1
            push!(Is,i);
            push!(Js,j);
            push!(Vs,1);
        elseif n*r - 1 < dist < n*r + 1
            push!(Is,i);
            push!(Js,j);
            push!(Vs,quadgk(t-> max(min(1.,t+n*r-dist),0) -min(max(0.,t-n*r-dist),1) ,0,1)[1]);
        end
    end
    return sparse(Is, Js, Vs)
end

end # end module
