"""
    EulerMaruyamaFPT(u₀, in_set::TIN, f::Tf, K, S, Q_sampler::TQ, Δt, maxΔt; nsave=1, verbose=false, save_trajectory=true) where {TIN,Tf,TQ}

`EulerMaruyamaFPT`: Integrate the Generalized Stochastic Kuramoto model:

du(i,t) = f(t,u(i,t))dt + ∑K(i,j)S(u(i,t),u(j,t))dt + dW

with Euler-Maryuama time stepping and terminate when a first passage is
triggered by function `in_set`.

### Fields
* `u₀` - Initial condition
* `in_set` - Boolean function which determines if we have left thet set from
  which we are studying the first passage time problem.
* `f` - f = f(t,u), self interaction term
* `K` - Connectivity matrix with weights, either dense or CSC format
* `S` - S = S(u(x), u(y)), interaction term
* `Q_sampler` - Sample the Q process assocaited with W(t) ∼ N(0,t*Q)
* `Δt` - Time step
* `maxΔt` - Maximum number of time steps
### Optional Fields
* `nsave = 1` - Number of time steps between recording
* `verbose = false` - Print when saving
"""
function EulerMaruyamaFPT(u₀, in_set::TIN, f::Tf, K, S, Q_sampler::TQ, Δt, maxΔt; nsave=1, verbose=false, save_trajectory=true) where {TIN,Tf,TQ}

    t = 0.0
    u = copy(u₀)
    t_trajectory = Float64[]
    u_trajectory = typeof(u)[]
    if save_trajectory
        push!(t_trajectory, t)
        push!(u_trajectory, copy(u))
    end

    # local terms
    local_force = zeros(length(u))
    # nonlocal terms
    nonlocal_force = zeros(length(u))
    # preallocate space for W(t) process
    ΔW = zeros(length(u))

    for k in 1:maxΔt

        # compute forces
        @. local_force = f(t, u)
        tabulate_nonlocal_force!(nonlocal_force, K, S, u)

        # generate ΔW
        ΔW .= sqrt(Δt) * Q_sampler()

        @. u += Δt * local_force + Δt * nonlocal_force + ΔW

        t += Δt
        if (mod(k, nsave) == 0)
            if verbose
                println(@sprintf("Saving at t = %g", t))
            end
            if (save_trajectory)
                push!(t_trajectory, t)
                push!(u_trajectory, copy(u))
            end
            if (!in_set(u))
                if verbose
                    println(@sprintf("First exit at t = %g", t))
                end
                # save the exit if we have not been saving the full trajectory
                if (!save_trajectory)
                    push!(t_trajectory, t)
                    push!(u_trajectory, copy(u))
                end

                break
            end
        end
    end

    if (in_set(u))
        println(@sprintf("NO EXIT: t = %g", t))
        # save the terminal state if we have not exited and we have not been
        # saving the full trajectory
        if (!save_trajectory)
            push!(t_trajectory, t)
            push!(u_trajectory, copy(u))
        end
    end

    return t_trajectory, u_trajectory
end

