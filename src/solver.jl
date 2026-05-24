# src/solver.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# THE USER-FACING INTERFACE TO THE SOLVER
#
# This file contains:
#
#   LyapunovState            — struct holding the full augmented state
#                              and total elapsed time, used for restarts.
#
#   lyapunov_exponents       — single run for one initial condition.
#                              Accepts either an x0 vector (fresh start)
#                              or a LyapunovState (warm restart from a
#                              previous computation).
#
#   lyapunov_exponents_mean  — convenience wrapper that repeats the single
#                              run over a list of initial conditions and
#                              returns mean and standard deviation.
# ─────────────────────────────────────────────────────────────────────────────

using OrdinaryDiffEq


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURE: LyapunovState
#  Holds the full augmented state at the end of an integration, plus the
#  total elapsed time. Used to restart a computation without starting over.
# ══════════════════════════════════════════════════════════════════════════════

"""
    LyapunovState(u, T_total)

Holds the result of a `lyapunov_exponents` run in a form that can be
used to restart the computation later.

# Fields
- `u::Vector{Float64}`  : full augmented state vector at end of integration
                          layout: [x(T); vec(E(T)); Λ(T)]
- `T_total::Float64`    : total integration time accumulated so far

# Usage
```julia
# First run — not yet converged
λ, state = lyapunov_exponents(prob, x0, T1; return_state=true)

# Continue from where we left off
λ, state = lyapunov_exponents(prob, state, T2)

# Exponents are now Λ(T1+T2) / (T1+T2)
```
"""
struct LyapunovState
    u       :: Vector{Float64}   # full augmented state [x; vec(E); Λ]
    T_total :: Float64           # total integration time accumulated so far
end


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION: lyapunov_exponents
#  Steps 2 + 3 of the workflow: solve the augmented system, return Λ(T)/T.
#  Accepts either a fresh x0 or a LyapunovState for warm restarts.
# ══════════════════════════════════════════════════════════════════════════════

"""
    lyapunov_exponents(prob, start, T; solver, reltol, abstol, return_state)

Integrate the augmented CGS system and return the k Lyapunov exponents
λ₁ ≥ λ₂ ≥ … ≥ λₖ.

# Arguments
- `prob::LyapunovProblem`              : the system definition
- `start::Vector{Float64}`             : initial condition x₀ for a fresh run
  OR
  `start::LyapunovState`               : saved state from a previous run,
                                         for continuing an integration
- `T::Float64`                         : additional integration time

# Keyword arguments
- `solver`        : ODE solver (default: Tsit5(), equivalent to ode45)
- `reltol`        : relative tolerance (default: 1e-6)
- `abstol`        : absolute tolerance (default: 1e-6)
- `return_state`  : if true, also return a `LyapunovState` for future restarts
                    (default: false)

# Returns
- `λ::Vector{Float64}` of length k                    if `return_state=false`
- `(λ::Vector{Float64}, state::LyapunovState)`        if `return_state=true`

# Examples
```julia
# Fresh run
λ = lyapunov_exponents(prob, x0, 1000.0)

# Fresh run, save state for possible restart
λ, state = lyapunov_exponents(prob, x0, 1000.0; return_state=true)

# Warm restart — continue from saved state for 500 more time units
# Exponents are now Λ(1500) / 1500
λ, state = lyapunov_exponents(prob, state, 500.0)
```
"""
function lyapunov_exponents(prob::LyapunovProblem,
                             start,           # Vector{Float64} or LyapunovState
                             T::Float64;
                             solver       = Tsit5(),
                             reltol       = 1e-6,
                             abstol       = 1e-6,
                             return_state = false)

    # ── Build initial state and track total elapsed time ──────────────────────
    # Dispatch on the type of `start`:
    #   Vector{Float64}  → fresh run, build u0 from x0
    #   LyapunovState    → warm restart, reuse saved u and accumulate T
    if isa(start, Vector{Float64})
        # Fresh run: build the full augmented initial state from x0
        u0      = make_initial_state(prob, start)
        T_total = T   # total time is just this run

    elseif isa(start, LyapunovState)
        # Warm restart: reuse the saved augmented state
        # The frame E and accumulators Λ carry over from the previous run.
        # The accumulators continue accumulating from where they left off,
        # so dividing by T_total at the end gives the correct time average.
        u0      = start.u
        T_total = start.T_total + T   # add new time to previous total

    else
        error("start must be a Vector{Float64} (fresh run) or LyapunovState (restart)")
    end

    # ── Build and solve the augmented ODE ────────────────────────────────────
    ode!     = build_ode!(prob)
    tspan    = (0.0, T)              # always integrate for T more time units
    ode_prob = ODEProblem(ode!, u0, tspan, nothing)

    sol = solve(ode_prob, solver;
                reltol         = reltol,
                abstol         = abstol,
                maxiters       = 1_000_000,
                save_everystep = false)

    # ── Extract results ───────────────────────────────────────────────────────
    # Λ(T) lives in the last k elements of the final state vector.
    # Dividing by T_total (not just T) gives the correct time average
    # when restarting — the accumulators contain the sum over ALL time.
    d       = prob.d
    k       = prob.k
    u_final = sol.u[end]
    Λ       = u_final[d + d*k + 1 : d + d*k + k]
    λ       = Λ ./ T_total

    # ── Return results ────────────────────────────────────────────────────────
    if return_state
        # Pack the final augmented state and total elapsed time into a
        # LyapunovState struct for future restarts.
        state = LyapunovState(u_final, T_total)
        return λ, state
    else
        return λ
    end
end


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTION: lyapunov_exponents_mean
# ══════════════════════════════════════════════════════════════════════════════

"""
    lyapunov_exponents_mean(prob, x0_list, T; kwargs...)

Run `lyapunov_exponents` for each initial condition in `x0_list` and
return the mean and standard deviation across all runs.

Keyword arguments (solver, reltol, abstol) are forwarded to
`lyapunov_exponents` for every run. Note: `return_state` is not
forwarded — use `lyapunov_exponents` directly if you need states.

# Returns
- `λ_mean::Vector{Float64}` — mean of each exponent across all runs
- `λ_std::Vector{Float64}`  — standard deviation (= rms deviation in paper)
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
