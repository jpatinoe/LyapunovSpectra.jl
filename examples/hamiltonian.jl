# examples/hamiltonian.jl
#
# Reproduces the Hamiltonian results from Christiansen & Rugh (1997):
#   - Figure 2: positive Lyapunov exponents from a single run
#   - Table  2: Mean and RMS deviation over 1000 runs
#
# Run with:
#   julia --project=. examples/hamiltonian.jl

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using LyapunovSpectra
using Random
using Statistics
using Printf
using LinearAlgebra

# ── Helper: compute Hamiltonian energy ───────────────────────────────────────
# MATLAB note: helper functions are defined at the top of the script in Julia.
# In MATLAB you would put them at the bottom as local functions.
function hamiltonian_energy(u)
    x, y, z    = u[1], u[2], u[3]
    px, py, pz = u[4], u[5], u[6]
    return (px^2 + py^2 + pz^2)/2 +
           x^2*y^2 + y^2*z^2 + z^2*x^2 +
           (x^4 + y^4 + z^4)/32
end

# ── Helper: generate initial conditions in a safe energy window ───────────────
# MATLAB note: this is the cleanest way to handle scoping in Julia scripts —
# wrap anything with loops and counters inside a function.
function generate_initial_conditions(n, rng)
    x0_list = Vector{Vector{Float64}}()
    base    = [0.5, 0.3, 0.7, 0.3, 0.2, 0.4]

    while length(x0_list) < n
        x0 = 0.05 .* randn(rng, 6) .+ base
        E  = hamiltonian_energy(x0)
        if 0.2 < E < 0.8       # ← safe energy window for this base point
            push!(x0_list, x0)
        end
    end
    return x0_list
end

println("="^60)
println("  Quartic Hamiltonian — Christiansen & Rugh (1997) Table 2")
println("="^60)

# ── Parameters ────────────────────────────────────────────────────────────────
T      = 10000.0
n_runs = 10

# Build the problem struct (d=6, k=6, β=0.5)
prob = hamiltonian_system()

# ── Single run (reproduces Figure 2) ─────────────────────────────────────────
rng_single = Random.MersenneTwister(42)
x0_single = [0.5, 0.3, 0.7, 0.3, 0.2, 0.4]
E_single   = hamiltonian_energy(x0_single)

println("\nSingle run (reproduces Figure 2 of the paper):")
println("  Initial condition: x0 = $x0_single")
@printf "  Energy: H = %.4f\n" E_single
println("  Integrating for T = $T ...")

λ_single = lyapunov_exponents(prob, x0_single, T)

# Symplectic structure check:
# For a Hamiltonian system exponents come in pairs (λₖ, -λₖ)
# so λₖ + λ_{7-k} should be exactly zero.
# This is our main validation — equivalent to the sum=-13.6667 check for Lorenz.
println("\n  Full spectrum:")
println("  ┌─────┬────────────┬─────────────────────────────┐")
println("  │  k  │    λₖ      │  symplectic check           │")
println("  ├─────┼────────────┼─────────────────────────────┤")
for k in 1:6
    pair_k   = 7 - k
    pair_sum = λ_single[k] + λ_single[pair_k]
    @printf "  │  %d  │  %+.4f  │  λ%d + λ%d = %+.6f      │\n" k λ_single[k] k pair_k pair_sum
end
println("  └─────┴────────────┴─────────────────────────────┘")

println("\n  Positive exponents only (paper Table 2, T=10000):")
@printf "  λ₁ = %+.4f  (paper: +0.2374)\n" λ_single[1]
@printf "  λ₂ = %+.4f  (paper: +0.1184)\n" λ_single[2]
@printf "  λ₃ = %+.4f  (paper: +0.000390)\n" λ_single[3]

# ── n_runs average (reproduces Table 2) ──────────────────────────────────────
println("\n$(n_runs)-run average (reproduces Table 2 of the paper):")

rng_multi = Random.MersenneTwister(0)

println("  Generating $n_runs initial conditions in energy window E ∈ (3, 6) ...")
x0_list = generate_initial_conditions(n_runs, rng_multi)

# Print energies so we can verify the filter worked
energies = hamiltonian_energy.(x0_list)   # MATLAB note: the dot broadcasts
                                           # hamiltonian_energy over the list,
                                           # like arrayfun in MATLAB
@printf "  Energy range: [%.3f, %.3f]\n" minimum(energies) maximum(energies)

println("  Running $n_runs integrations of length T=$T ...")
println("  (each run is a 6D system — this will take a few minutes)\n")

elapsed = @elapsed begin
    λ_mean, λ_std = lyapunov_exponents_mean(prob, x0_list, T)
end

# Print full spectrum
println("  Full spectrum results:")
println("  ┌─────┬────────────┬────────────┐")
println("  │  k  │    λₖ      │  rms dev   │")
println("  ├─────┼────────────┼────────────┤")
for k in 1:6
    @printf "  │  %d  │  %+.4f  │  %.2e  │\n" k λ_mean[k] λ_std[k]
end
println("  └─────┴────────────┴────────────┘")

# Compare positive exponents against Table 2
println("\n  Positive exponents vs paper (Table 2):")
println("  ┌─────┬────────────┬────────────┬────────────┐")
println("  │  k  │    λₖ      │  rms dev   │  paper λₖ  │")
println("  ├─────┼────────────┼────────────┼────────────┤")
@printf "  │  1  │  %+.4f  │  %.2e  │  +0.2374   │\n" λ_mean[1] λ_std[1]
@printf "  │  2  │  %+.4f  │  %.2e  │  +0.1184   │\n" λ_mean[2] λ_std[2]
@printf "  │  3  │  %+.4f  │  %.2e  │  +0.000390 │\n" λ_mean[3] λ_std[3]
println("  └─────┴────────────┴────────────┴────────────┘")

# Symplectic check on the averages
println("\n  Symplectic pair sums (should all be ~0):")
for k in 1:3
    pair_k   = 7 - k
    pair_sum = λ_mean[k] + λ_mean[pair_k]
    @printf "  λ%d + λ%d = %+.6f\n" k pair_k pair_sum
end

@printf "\n  Total wall time: %.1f seconds\n" elapsed