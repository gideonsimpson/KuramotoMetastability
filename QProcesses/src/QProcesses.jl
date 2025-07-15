module QProcesses

using FFTW
using Random

"""
`NullProcess`: Returns zero

### Fields
* `n` - Number of sample points
"""
function NullProcess(n)
    return zeros(n)
end

"""
`IsoGaussian`: Samples the Iso-Gaussian N(0,σ²I) on n sample points

### Fields
* `σ` - Intensity
* `n` - Number of sample points
"""
function IsoGaussian(σ,n)
    return σ * randn(n)
end

"""
`ComplexInvLaplacianPBC`: Sample a periodic complex valued path on [0,X) with n
sample points, N(0, σ²C₀).  The covariance operator is C₀ = (-Δ)^{-s/2}.

### Fields
* `σ` - Intensity
* `X` - Domain [0,X)
* `n` - Number of sample points
### Optional Fields
* `s` - Regularity, s=2
"""
function ComplexInvLaplacianPBC(σ, X, n; s=2)
    k = [0:(n÷2); -(n÷2)+1:1:-1];
    λ= zeros(n);

    uhat = zeros(ComplexF64,n)
    ξ = randn(n);
    @. λ[2:end] = abs(X/(2*π * k[2:end]))^(s);
    @. uhat = σ * n * sqrt(λ) * ξ /sqrt(X) ;
    u = ifft(uhat);
    return u;
end

"""
`RealInvLaplacianPBC`: Sample a periodic real valued path on [0,X) with n
sample points, N(0, σ²C₀).  The covariance operator is C₀ = (-Δ)^{-s/2}.

### Fields
* `σ` - Intensity
* `X` - Domain [0,X)
* `N` - Number of sample points
### Optional Fields
* `s` - Regularity, s=2
"""
function RealInvLaplacianPBC(σ, X, n; s=2)
    k = [0:(n÷2); -(n÷2)+1:1:-1];
    λ= zeros(n);

    uhat = zeros(ComplexF64,n)
    ξ = randn((n÷2)-1);
    η = randn((n÷2)-1);

    χ = zeros(ComplexF64,n);
    @. χ[2:(n÷2)] = (ξ + im * η)/sqrt(2);
    @. χ[(n÷2)+2:end] = conj(χ[(n÷2):-1:2] );

    @. λ[2:end] = abs(X/(2*π * k[2:end]))^(s);
    λ[(n÷2)+1]=0;

    @. uhat = σ * n * sqrt(λ) * χ /sqrt(X);
    u = ifft(uhat);
    return real.(u);
end

"""
`RealInvLaplacianNeumann`: Sample a real valued path on [0,X] with n+1
sample pointsm, N(0, σ²C₀).  The covariance operator is C₀ = (-Δ)^{-s/2} with
Neumann BCs.

### Fields
* `σ` - Intensity
* `X` - Domain [0,X)
* `nN` - Number of sample points
### Optional Fields
* `s` - Regularity, s=2
"""
function RealInvLaplacianNeumann(σ, X, n; s=2)
    k = [0:n; -n+1:1:-1];
    λ= zeros(2*n);

    uhat = zeros(ComplexF64,2*n)
    ξ = randn(n);

    @. λ[2:end] = abs(X/(π * k[2:end]))^(s);
    @. uhat[2:n+1] = σ * 2*n * sqrt(λ[2:n+1]) * ξ* sqrt(2/X);
    u = real.(ifft(uhat));
    return [u[n+1:end]; u[1]]
end

"""
`RealInvLaplacianDirichlet`: Sample a real valued path on (0,X) with n-1
sample points, N(0, σ²C₀).  The covariance operator is C₀ = (-Δ)^{-s/2} with
Dirichlet BCs.

### Fields
* `σ` - Intensity
* `X` - Domain [0,X)
* `n` - Number of sample points
### Optional Fields
* `s` - Regularity, s=2
"""
function RealInvLaplacianDirichlet(σ, X, n; s=2)
    k = [0:n; -n+1:1:-1];
    λ= zeros(2*n);

    uhat = zeros(ComplexF64,2*n)
    ξ = randn(n);
    @. λ[2:end] = abs(X/(π * k[2:end]))^(s);

    @. uhat[2:n+1] = σ*2*n * sqrt(λ[2:n+1]) * ξ* sqrt(2/X);
    u = imag.(ifft(uhat))[n+2:end];
    return u

end

"""
`QCholesky` - Sample from the Gaussian N(0,Q) where Q = LLᵀ has been factored

### Fields
* `L` - Lower triangular matrix from the Cholesky factorization of Q
"""
function QCholesky(L)
    n = size(L)[1];
    return L * randn(n)
end

function DirichletLaplacianQPWC1D(n)
    #=
    Approximate the Q matrix for (-Δ)\^{-1} over [0,1] with Dirichlet BCs for
    piecewise constant Galerkin FEMs.
    =#

    Q = zeros(n,n);

    Δx = 1.0/n;

    for i =1:n
        for j = 1:n
            xᵢ = i * Δx;
            xⱼ = j * Δx;
            if i<j
                Q[i,j] = Δx^2 * (xᵢ - Δx/2) * (1 - xⱼ + Δx/2);
            elseif i>j
                Q[i,j] = Δx^2 * (xⱼ - Δx/2) * (1 - xᵢ + Δx/2);
            else
                Q[i,j] = Δx^2 * (xᵢ- xᵢ^2 - 2/3 * Δx + xᵢ * Δx - Δx^2/4);
            end

        end
    end

    return Q

end

end
