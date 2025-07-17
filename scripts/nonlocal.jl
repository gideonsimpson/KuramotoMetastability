
using Printf
using Random
using LinearAlgebra
using ForwardDiff
using StringMethod
using JLD2
using DSP

modulepath = "/path/to/KuramotoMetastability";

push!(LOAD_PATH, modulepath * "/GraphMatrices");
push!(LOAD_PATH, modulepath * "/TwistedStates");
push!(LOAD_PATH, modulepath * "/ClassicalKuramoto");

using GraphMatrices
using TwistedStates
using ClassicalKuramoto

@show k = 2;                    # interaction range, k>1 is nonlocal
@show n_vals = 5 * 2 .^ (3:1:8);# number of sites to simulate
@show q = 4;                    # initial twisted state
@show n_beads = 200;            # number of points to use in the string method
@show Δt = 1e-2;                # algorithmic time step for string method
flush(stdout); 

α = 0;                          # attractive interactions
pin = true                      # pin the endpoints of the string

E = u -> ClassicalKuramoto.energy(u, k, α);
gradE = u -> ForwardDiff.gradient(E, u);
V = w -> ClassicalKuramoto.energy([0; w], k, α)

function get_saddle(U, Δt)
    n_beads = length(U)
    n = length(U[1]) + 1

    function ∇V_modifed(w)
        gradVext = zeros(n)
        ClassicalKuramoto.grad_energy!(gradVext, [0; w], k, α)
        @. gradVext -= gradVext[1]
        return gradVext[2:end]
    end

    dist = (u, v) -> norm(u .- v, 2)
    string = SimplifiedString(∇V_modifed, stepRK4!,
        spline_reparametrize!, dist, pin, Δt)
    opts = StringOptions(verbose=true, save_trajectory=false,
        nmax=10^7, print_iters=10^4, tol=1.e-6)

    println("running string method")
    flush(stdout);
    StringMethod.simplified_string!(U, string, options=opts)
    flush(stdout);

    E_vals = [V(U[i]) for i in 1:n_beads]
    idx = argmax(E_vals)
    τ = upwind_tangent(U[idx-1:idx+1], E)

    climb = ClimbingImage(∇V_modifed, τ, stepRK4!, Δt)
    opts = StringMethod.SaddleOptions(verbose=true, save_trajectory=false,
        print_iters=10^4, nmax=10^7, tol=1e-8)
    println("running climbing image method")
    flush(stdout);
    saddle_trajectory = climbing_image(U[idx], climb, options=opts)
    flush(stdout);

    return U, [0; saddle_trajectory[end]]
end

string_paths = [];
saddle_states = [];
for n in n_vals
    @show n
    u_left = unwrap(construct_q_twisted(n, q),range=1)
    u_right = unwrap(construct_q_twisted(n, q - 1, invert=false),range=1)
    U = linear_string(u_left[2:end], u_right[2:end], n_beads)
    U_, saddle_ = get_saddle(U, Δt)
    push!(string_paths, deepcopy(U_))
    push!(saddle_states, deepcopy(saddle_))
    flush(stdout);
end


fname = replace(@sprintf("nonlocal_q%d_k%d_nmin%d_nmax%d_dt%g_N%d",
        q, k, minimum(n_vals), maximum(n_vals),Δt,n_beads), "." => "_");

jldsave(string(fname, ".jld2"); n_vals, k, q, Δt, n_beads, α, string_paths, saddle_states);
@printf("saved to %s.jld2\n", fname);
