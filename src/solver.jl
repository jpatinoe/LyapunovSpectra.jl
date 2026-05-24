# src/solver.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# THIS IS THE USER INTERFACE.
#
# This file contains two functions:
#
#   lyapunov_exponents       — single run for one initial condition.
#                              solve the augmented system, then extract λ = Λ/T.
#
#   lyapunov_exponents_mean  — this is just a convenience wrapper that repeats the single
#                              run over a list of initial conditions and
#                              returns mean and standard deviation. For testing and statistics,
#                              not essential to the algorithm.
#
# This file is purely about calling the solver and packaging the output.
# ─────────────────────────────────────────────────────────────────────────────

using OrdinaryDiffEq   # ODEProblem, solve(), Tsit5()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION: lyapunov_exponents
# ══════════════════════════════════════════════════════════════════════════════

"""
    lyapunov_exponents(prob, x0, T; solver, reltol, abstol)

Integrate the augmented CGS (Continuous Gram-Schmidt) system from t=0 to t=T and return the
k Lyapunov exponents λ₁ ≥ λ₂ ≥ … ≥ λₖ.

# Arguments
- `prob::LyapunovProblem` : the system definition (from algorithm.jl)
- `x0::Vector{Float64}`   : initial condition for the trajectory only
                            (the frame and accumulators are initialised
                             automatically by make_initial_state)
- `T::Float64`            : total integration time

# Keyword arguments (optional — sensible defaults provided)
- `solver`  : ODE solver object. Default: Tsit5(), a modern 4th/5th order
              Runge-Kutta method.
              Could be swapped for any OrdinaryDiffEq solver (with caution).
- `reltol`  : relative tolerance for the ODE solver. Default: 1e-6.
- `abstol`  : absolute tolerance for the ODE solver. Default: 1e-6.

# Returns
- `λ::Vector{Float64}` of length k — the estimated Lyapunov exponents,
  in descending order λ₁ ≥ λ₂ ≥ … ≥ λₖ.
"""
function lyapunov_exponents(prob::LyapunovProblem,
                             x0::Vector{Float64},
                             T::Float64;
                             solver  = Tsit5(),   # ← swap here with caution for a different integrator
                             reltol  = 1e-6,
                             abstol  = 1e-6)

    # ── Build the initial state vector u₀ ─────────────────────────────────────
    # x₀ only has d components (the trajectory). The full augmented state u₀
    # has d + d*k + k components (trajectory + frame + accumulators).
    # make_initial_state assembles u₀ = [x₀; vec(E₀); zeros(k)].
    u0 = make_initial_state(prob, x0)

    # ── Build the augmented ODE right-hand side ────────────────────────────────
    # build_ode! assembles the full augmented system (eq. 6) automatically from prob.v, prob.J, prob.β.
    # The returned ode! function encodes all d + d*k + k equations.
    ode! = build_ode!(prob)

    # ── Set up the ODE problem ─────────────────────────────────────────────────
    # ODEProblem packages the ODE function, initial condition, and time span
    # into one object, without solving anything yet.
    # tspan = (0.0, T) means: integrate from t=0 to t=T.
    # The last argument `nothing` is the parameter vector p — unused because
    # all parameters are captured inside the ode! closure.
    tspan    = (0.0, T)
    ode_prob = ODEProblem(ode!, u0, tspan, nothing)

    # ── Solve the augmented ODE system ────────────────────────────────────────
    # This is the actual solve call. `solver` defaults to Tsit5() —
    # a modern explicit Runge-Kutta method with adaptive step size,
    #
    # save_everystep = false: only store the final state u(T), not the
    # full trajectory. We only need Λ(T) at the end, so saving every
    # step would waste memory for no benefit.
    #
    # maxiters = 1_000_000: maximum number of time steps before the solver
    # gives up.
    sol = solve(ode_prob, solver;
                reltol         = reltol,
                abstol         = abstol,
                maxiters       = 1_000_000,
                save_everystep = false)

    # ── Extract results ───────────────────────────────────────
    # sol.u[end] is the final state vector u(T) — a flat array of length
    # d + d*k + k. We only need the last k elements: the accumulators Λ(T).
    #
    # The accumulators start at position d + d*k + 1 in the flat array:
    #   positions 1   … d       → x(T)    (trajectory, discarded)
    #   positions d+1 … d+d*k   → E(T)    (frame, discarded)
    #   positions d+d*k+1 … end → Λ(T)    (accumulators — what we want)
    d       = prob.d
    k       = prob.k
    u_final = sol.u[end]
    Λ       = u_final[d + d*k + 1 : d + d*k + k]

    # Lyapunov exponents = Λₘ(T) / T  (paper eq. 7, Theorem p.1065)
    return Λ ./ T
end


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRA FUNCTION: lyapunov_exponents_mean
#
#  Repeats lyapunov_exponents over a list of initial conditions and
#  computes statistics. Not essential to the algorithm — just a loop with
#  mean and std at the end.
# ══════════════════════════════════════════════════════════════════════════════

"""
    lyapunov_exponents_mean(prob, x0_list, T; kwargs...)

Run `lyapunov_exponents` for each initial condition in `x0_list` and
return the mean and standard deviation across all runs.

Keyword arguments (solver, reltol, abstol) are forwarded to
`lyapunov_exponents` for every run.

# Returns
- `λ_mean::Vector{Float64}` — mean of each exponent across all runs
- `λ_std::Vector{Float64}`  — standard deviation (= rms deviation in paper)
"""
function lyapunov_exponents_mean(prob::LyapunovProblem,
                                  x0_list::Vector{Vector{Float64}},
                                  T::Float64; kwargs...)

    # ── Run lyapunov_exponents for every initial condition ────────────────────
    # This is a list comprehension.
    # `kwargs...` forwards all keyword arguments (solver, reltol, etc.)
    # transparently to each lyapunov_exponents call.
    results = [lyapunov_exponents(prob, x0, T; kwargs...) for x0 in x0_list]

    # ── Stack results into a matrix and compute statistics ────────────────────
    # results is a vector of k-element vectors (one per run).
    # hcat(results...) stacks them as columns → d×n_runs matrix.
    # Transposing (') gives an n_runs×k matrix where:
    #   row i   = the k exponents from run i
    #   column m = all estimates of λₘ across runs
    result_matrix = hcat(results...)'   # n_runs × k matrix

    # Compute mean and std along dimension 1 (across rows = across runs),
    # giving one mean and one std per exponent.
    # vec() converts the 1×k result of mean/std into a plain length-k vector.
    λ_mean = vec(mean(result_matrix, dims=1))
    λ_std  = vec(std(result_matrix,  dims=1))

    return λ_mean, λ_std
end
