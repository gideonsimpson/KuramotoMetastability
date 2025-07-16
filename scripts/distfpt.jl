using Distributed, SlurmClusterManager
using Random
using Printf

addprocs(SlurmManager())
@show nprocs();
@show gethostname();

n_samples = 10^2;
seed = 1000;

@everywhere n = 10; # number of sites
@everywhere k = 1; # interaction rage
@everywhere q = 1; # initial twisted state, 
@everywhere Δt = 1e-2;
@everywhere β = 10.0;
@everywhere α = 0;
@everywhere X = 1.0;
@everywhere tmax = 10^6;
@everywhere Δt_save = Δt;

@show n;
@show k;
@show q;
@show Δt;
@show β;

# picotte
@everywhere modulepath = "/path/to/KuramotoMetastability";

@everywhere push!(LOAD_PATH, modulepath*"/GeneralizedKuramoto");
@everywhere push!(LOAD_PATH, modulepath*"/QProcesses");
@everywhere push!(LOAD_PATH, modulepath*"/GraphMatrices");
@everywhere push!(LOAD_PATH, modulepath*"/TwistedStates");
@everywhere push!(LOAD_PATH, modulepath*"/ClassicalKuramoto");

@everywhere begin
    using Printf
    using Random
    using DSP
    using Statistics
    using Optimization
    using OptimizationOptimJL
    using JLD2

    using GeneralizedKuramoto
    using QProcesses
    using GraphMatrices
    using TwistedStates
    using ClassicalKuramoto
end

@everywhere begin

    # prepare data
    u₀ = mod.(TwistedStates.construct_q_twisted(n, q), 1)

    maxΔt = Int(tmax / Δt);
    nsave = Int(Δt_save / Δt);
    
    # prepare interaction matrix
    K = 1 / (2 * k + 1) * GraphMatrices.discrete_knn_spmatrix(n, k);
    # K = GraphMatrices.discrete_knn_spmatrix(n, k)

    σ = sqrt(2/β);
    # prepare sampler
    Q_sampler = let σ = σ, n = n
        () -> QProcesses.IsoGaussian(σ, n);
    end
    # prepare nonlinearity
    f = (t, u) -> 0.0
    # coupling
    S = let α = α
        (u, v) -> (-1)^(α) * sinpi(2 * (v - u));
    end

    # energy function
    E = u -> ClassicalKuramoto.energy(u, K, α);

    # optimization function
    opt_func = OptimizationFunction((v, p) -> E([0; v]), Optimization.AutoForwardDiff())
    opt_prob = OptimizationProblem(opt_func, u₀[2:end])
    soln = solve(opt_prob, BFGS())
    Einit = soln.objective

    function in_well(u; tol=1e-6)
        opt_prob.u0 .= unwrap(u .- u[1], range=1)[2:end]
        soln = solve(opt_prob, BFGS())
        return Bool(abs(soln.objective - Einit) < tol)
    end

    function sample_fpt()
        t_trajectory, u_trajectory = GeneralizedKuramoto.EulerMaruyamaFPT(u₀, in_well, f, K,
            S, Q_sampler, Δt, maxΔt, nsave=nsave, verbose=false, save_trajectory=false)
        opt_prob.u0 .= unwrap(u_trajectory[end] .- u_trajectory[end][1], range=1)[2:end]
        soln = solve(opt_prob, BFGS())
        return t_trajectory[end], soln.objective, u_trajectory[end]
    end

end

Random.seed!(seed);
samples = pmap(idx -> sample_fpt(), 1:n_samples);

t_exit_values = [sample[1] for sample in samples];
E_exit_values = [sample[2] for sample in samples];
u_exit_values = [sample[3] for sample in samples];

fname = replace(@sprintf("fpt2_n%d_q%d_k%d_N%d_beta%g_dt%g_tmax%g_s%d",
        n, q, k, n_samples, β, Δt, tmax, seed), "." => "_");

jldsave(string(fname, ".jld2"); n, k, q, Δt, σ, β, α, X, tmax, Δt_save, t_exit_values, E_exit_values, u_exit_values, seed);
@printf("saved to %s.jld2\n", fname);
