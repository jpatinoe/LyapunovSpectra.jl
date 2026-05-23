# examples/lorenz.jl
#
# Reproduces the Lorenz results from Christiansen & Rugh (1997):
#   - Figure 1: Lyapunov exponents from a single run
#   - Table  1: Mean and RMS deviation over 1000 runs
#
# Run this script from the terminal with:
#   julia --project=. examples/lorenz.jl
# ─────────────────────────────────────────────────────────────────────────────
# Tell Julia where to find our package source
# MATLAB equivalent: addpath('../src')
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using LyapunovSpectra
using Random
using Statistics
using Printf 

println("="^60)
println("  Lorenz System — Christiansen & Rugh (1997) Table 1")
println("="^60)

# ── Parameters ────────────────────────────────────────────────────────────────
# These match the paper exactly:
#   T       = 1000  integration time per run
#   n_runs  = 1000  number of random initial conditions
#   β       = 20    stability parameter (chosen in section 3)
# ─────────────────────────────────────────────────────────────────────────────
T      = 1000.0
n_runs = 1000

# Build the problem struct (d=3, k=3, β=20)
prob = lorenz_system()

# ── Single run (reproduces Figure 1) ─────────────────────────────────────────
rng_single = Random.MersenneTwister(42)

# A typical initial condition 
x0_single = [1.0, 0.0, 0.0]

println("\nSingle run (reproduces Figure 1 of the paper):")
println("  Initial condition: x0 = $x0_single")
println("  Integrating for T = $T ...")

λ_single = lyapunov_exponents(prob, x0_single, T)

# The format string uses %+.4f to always show the sign.
@printf "  λ₁ = %+.4f  (paper: +0.9057)\n"  λ_single[1]
@printf "  λ₂ = %+.4f  (paper: ~0.0000)\n"  λ_single[2]
@printf "  λ₃ = %+.4f  (paper: -14.5724)\n" λ_single[3]
@printf "  Sum = %.4f  (should equal -σ-1-b = -%.4f)\n" sum(λ_single) (10+1+8/3)

# ── 1000 runs (reproduces Table 1) ────────────────────────────────────────────
println("\n1000-run average (reproduces Table 1 of the paper):")
println("  Generating $n_runs random initial conditions ...")

# We use a fixed seed to get the same results every time we run.
rng_multi = Random.MersenneTwister(0)

# Generate n_runs random initial conditions near the Lorenz attractor.
# Each initial condition is a small random perturbation of [1, 0, 0].
x0_list = [randn(rng_multi, 3) .* [1.0, 1.0, 1.0] .+ [1.0, 0.0, 0.0]
           for _ in 1:n_runs]

println("  Running $n_runs integrations of length T=$T ...")
println("  (this will take a minute — 1000 × T=1000 is a lot of work)\n")

elapsed = @elapsed begin
    λ_mean, λ_std = lyapunov_exponents_mean(prob, x0_list, T)
end

# Print results in the same format as Table 1 of the paper
println("  Results:")
println("  ┌─────┬────────────┬────────────┬────────────┐")
println("  │  k  │    λₖ      │  rms dev   │  paper λₖ  │")
println("  ├─────┼────────────┼────────────┼────────────┤")
@printf "  │  1  │  %+.4f  │  %.2e  │  +0.9057   │\n" λ_mean[1] λ_std[1]
@printf "  │  2  │  %+.4f  │  %.2e  │  ~0.0000   │\n" λ_mean[2] λ_std[2]
@printf "  │  3  │  %+.4f  │  %.2e  │  -14.5724  │\n" λ_mean[3] λ_std[3]
println("  └─────┴────────────┴────────────┴────────────┘")
@printf "\n  Sum of exponents: %.4f  (should be -13.6667)\n" sum(λ_mean)
@printf "  Total wall time:  %.1f seconds\n" elapsed