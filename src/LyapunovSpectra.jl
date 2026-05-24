# src/LyapunovSpectra.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# THIS IS THE MODULE ROOT 
# ─────────────────────────────────────────────────────────────────────────────
module LyapunovSpectra

# ── External dependencies ─────────────────────────────────────────────────────

using LinearAlgebra      # dot(), norm(), mul!(), I (identity), qr()
using Random             # random number generation (used in examples/tests)
using Statistics         # mean(), std() — used in lyapunov_exponents_mean
using OrdinaryDiffEq     # ODEProblem, solve(), Tsit5() — the ODE solver ecosystem

# ── Include source files ───────────────────────────────────────────────────────
# Order matters: algorithm.jl must come first because
# systems.jl and solver.jl use types and functions defined in it.

include("algorithm.jl")   # Core: LyapunovProblem struct, build_ode!, make_initial_state
include("systems.jl")     # Example systems: Lorenz, quartic Hamiltonian
include("solver.jl")      # User interface: lyapunov_exponents, lyapunov_exponents_mean

# ── Exports ────────────────────────────────────────────────────────────────

export LyapunovProblem,         # the problem definition struct  (algorithm.jl)
       build_ode!,              # constructs the augmented ODE   (algorithm.jl)
       make_initial_state,      # builds the initial state vector (algorithm.jl)
       lyapunov_exponents,      # main solver function            (solver.jl)
       lyapunov_exponents_mean, # statistics over many runs       (solver.jl)
       lorenz_system,           # Lorenz system constructor        (systems.jl)
       hamiltonian_system       # Hamiltonian system constructor   (systems.jl)

end # module
