# src/solver.jl
#
# High-level solver: takes a LyapunovProblem + initial condition,
# integrates the augmented system, and returns Lyapunov exponents.

using OrdinaryDiffEq

"""
    lyapunov_exponents(prob, x0, T; solver, reltol, abstol)

Integrate the augmented CGS system from time 0 to T and return
the k Lyapunov exponents λ₁ ≥ λ₂ ≥ … ≥ λₖ.

# Arguments
- `prob::LyapunovProblem` : the system definition
- `x0::Vector{Float64}`   : initial condition for the trajectory
- `T::Float64`            : integration time

# Keyword arguments
- `solver`  : ODE solver (default: Tsit5())
- `reltol`  : relative tolerance (default 1e-6)
- `abstol`  : absolute tolerance (default 1e-6)

# Returns
- `λ::Vector{Float64}` of length k  — the Lyapunov exponents
"""
function lyapunov_exponents(prob::LyapunovProblem,
                             x0::Vector{Float64},
                             T::Float64;
                             solver  = Tsit5(),
                             reltol  = 1e-6,
                             abstol  = 1e-6)

    u0       = make_initial_state(prob, x0)   
    ode!     = build_ode!(prob)
    tspan    = (0.0, T)
    ode_prob = ODEProblem(ode!, u0, tspan, nothing)

    sol = solve(ode_prob, solver;
                reltol         = reltol,
                abstol         = abstol,
                maxiters       = 1_000_000,
                save_everystep = false)

    d       = prob.d
    k       = prob.k
    u_final = sol.u[end]
    Λ       = u_final[d + d*k + 1 : d + d*k + k]

    return Λ ./ T
end

"""
    lyapunov_exponents_mean(prob, x0_list, T; kwargs...)

Run `lyapunov_exponents` for each initial condition in `x0_list`
and return the mean and standard deviation across runs.

# Returns
- `λ_mean::Vector{Float64}` — mean of each exponent across runs
- `λ_std::Vector{Float64}`  — std deviation (rms deviation in the paper)
"""

function lyapunov_exponents_mean(prob::LyapunovProblem,
                                  x0_list::Vector{Vector{Float64}},
                                  T::Float64; kwargs...)

    results       = [lyapunov_exponents(prob, x0, T; kwargs...) for x0 in x0_list]
    result_matrix = hcat(results...)'

    λ_mean = vec(mean(result_matrix, dims=1))
    λ_std  = vec(std(result_matrix,  dims=1))

    return λ_mean, λ_std
end

