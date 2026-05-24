# src/LyapunovSpectra.jl
module LyapunovSpectra

using LinearAlgebra
using Random
using Statistics
using OrdinaryDiffEq

include("algorithm.jl")
include("systems.jl")
include("solver.jl")

export LyapunovProblem,         # problem definition struct    (algorithm.jl)
       LyapunovState,           # saved state for restarts     (solver.jl)
       build_ode!,              # constructs the augmented ODE (algorithm.jl)
       make_initial_state,      # builds the initial state     (algorithm.jl)
       lyapunov_exponents,      # main solver function         (solver.jl)
       lyapunov_exponents_mean, # statistics over many runs    (solver.jl)
       lorenz_system,           # Lorenz system constructor    (systems.jl)
       hamiltonian_system       # Hamiltonian system constructor (systems.jl)

end # module
