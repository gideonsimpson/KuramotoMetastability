"""
`Euler`: Integrate the Generalized Kuramoto model:

u̇(i,t) = f(t,u(i,t)) + ∑K(i,j)S(u(i,t),u(j,t))

with Euler time stepping.

### Fields
* `u₀` - Initial condition
* `f` - f = f(t,u), self interaction term
* `K` - Connectivity matrix with weights, either dense or CSC format
* `S` - S = S(u(x), u(y)), interaction term
* `Δt` - Time step
* `nΔt` - Total number of time steps
### Optional Fields
* `nsave = 1` - Number of time steps between recording
* `verbose = false` - Print when saving
"""
function Euler(u₀, f::Tf, K, S, Δt, nΔt; nsave=1, verbose=false) where {Tf}

    t = 0.0
    u = copy(u₀)
    t_trajectory = [t]
    u_trajectory = [copy(u)]

    # local terms
    local_force = zeros(length(u))
    # nonlocal terms
    nonlocal_force = zeros(length(u))

    for n in 1:nΔt

        # compute forces
        @. local_force = f(t, u)
        tabulate_nonlocal_force!(nonlocal_force, K, S, u)

        @. u += Δt * local_force + Δt * nonlocal_force

        t += Δt
        if (mod(n, nsave) == 0)
            if verbose
                println(@sprintf("Saving at t = %g", t))
            end
            push!(t_trajectory, t)
            push!(u_trajectory, copy(u))
        end
    end

    return t_trajectory, u_trajectory
end

"""
`EulerMaruyama`: Integrate the Generalized Stochastic Kuramoto model:

du(i,t) = f(t,u(i,t))dt + ∑K(i,j)S(u(i,t),u(j,t))dt + dW

with Euler-Maryuama time stepping.

### Fields
* `u₀` - Initial condition
* `f` - f = f(t,u), self interaction term
* `K` - Connectivity matrix with weights, either dense or CSC format
* `S` - S = S(u(x), u(y)), interaction term
* `Q_sampler` - Sample the Q process assocaited with W(t) ∼ N(0,t*Q)
* `Δt` - Time step
* `nΔt` - Total number of time steps
### Optional Fields
* `nsave = 1` - Number of time steps between recording
* `verbose = false` - Print when saving
"""
function EulerMaruyama(u₀, f::Tf, K, S, Q_sampler::TQ, Δt, nΔt; nsave=1, verbose=false) where {Tf,TQ}

    t = 0.0
    u = copy(u₀)
    t_trajectory = [t]
    u_trajectory = [copy(u)]

    # local terms
    local_force = zeros(length(u))
    # nonlocal terms
    nonlocal_force = zeros(length(u))
    # preallocate space for W(t) process
    ΔW = zeros(length(u))

    for k in 1:nΔt

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
            push!(t_trajectory, t)
            push!(u_trajectory, copy(u))
        end
    end

    return t_trajectory, u_trajectory
end


"""
`EulerMaruyamaGivenNoise`: Integrate the Generalized Stochastic Kuramoto model:

du(i,t) = f(t,u(i,t))dt + ∑K(i,j)S(u(i,t),u(j,t))dt + dW

with Euler-Maryuama time stepping with a given noise trajectory that has already
been generated.

### Fields
* `u₀` - Initial condition
* `f` - f = f(t,u), self interaction term
* `K` - Connectivity matrix with weights, either dense or CSC format
* `S` - S = S(u(x), u(y)), interaction term
* `Q_sampler` - Trajectory for the Q process associated with W(t) ∼ N(0,t*Q)
* `Δt` - Time step
* `nΔt` - Total number of time steps
### Optional Fields
* `nsave = 1` - Number of time steps between recording
* `verbose = false` - Print when saving
"""
function EulerMaruyama(u₀, f::Tf, K, S, Q_sampler::TQ, Δt, nΔt; nsave=1, verbose=false) where {Tf,TQ<:AbstractVector}

    t = 0.0
    u = copy(u₀)
    t_trajectory = [t]
    u_trajectory = [copy(u)]

    # local terms
    local_force = zeros(length(u))
    # nonlocal terms
    nonlocal_force = zeros(length(u))

    for k in 1:nΔt

        # compute forces
        @. local_force = f(t, u)
        tabulate_nonlocal_force!(nonlocal_force, K, S, u)

        @. u += Δt * local_force + Δt * nonlocal_force + sqrt(Δt) * Q_sampler[k]

        t += Δt
        if (mod(k, nsave) == 0)
            if verbose
                println(@sprintf("Saving at t = %g", t))
            end
            push!(t_trajectory, t)
            push!(u_trajectory, copy(u))
        end
    end

    return t_trajectory, u_trajectory
end

"""
`EulerMaruyamaObs`: Integrate the Generalized Stochastic Kuramoto model:

du(i,t) = f(t,u(i,t))dt + ∑K(i,j)S(u(i,t),u(j,t))dt + dW

with Euler-Maryuama time stepping evaluating a collection of observables on the
solution and returning their time series.

### Fields
* `u₀` - Initial condition
* `f` - f = f(t,u), self interaction term
* `K` - Connectivity matrix with weights, either dense or CSC format
* `S` - S = S(u(x), u(y)), interaction term
* `Q_sampler` - Sample the Q process assocaited with W(t) ∼ N(0,t*Q)
* `Δt` - Time step
* `nΔt` - Total number of time steps
### Optional Fields
* `nsave = 1` - Number of time steps between recording
* `verbose = false` - Print when saving
"""
function EulerMaruyamaObs(u₀, f::Tf, K, S, Q_sampler::TQ, Δt, nΔt, observables::Tuple{Vararg{<:Function,NO}}; nsave=1, verbose=false) where {Tf,TQ,NO}

    t = 0.0
    u = copy(u₀)

    # local terms
    local_force = zeros(length(u))
    # nonlocal terms
    nonlocal_force = zeros(length(u))
    # preallocate space for W(t) process
    ΔW = zeros(length(u))

    t_trajectory = Float64[t]

    observables_trajectory = zeros(NO, nΔt ÷ nsave + 1)
    obs_idx = 1
    ntuple(i -> observables_trajectory[i, obs_idx] = (observables[i])(u), NO)
    obs_idx += 1


    for k in 1:nΔt

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
            push!(t_trajectory, t)
            ntuple(i -> observables_trajectory[i, obs_idx] = (observables[i])(u), NO)
            obs_idx += 1
        end
    end

    return t_trajectory, observables_trajectory
end

