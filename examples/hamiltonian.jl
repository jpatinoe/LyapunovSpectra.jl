# examples/hamiltonian.jl
#
# Reproduces the Hamiltonian results from Christiansen & Rugh (1997):
# Note that the paper's Table 2 is based on n_runs=1000 runs, not just a single run.
# The results might differ because of the initial condition. (The paper doesn't specify how the initial condition was chosen)
#   - Figure 2: positive Lyapunov exponents from a single run
#   - Table  2: Mean and RMS deviation over n_runs runs
#
# Run with:
#   julia --project=. examples/hamiltonian.jl

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using LyapunovSpectra
using Random
using Statistics
using Printf
using LinearAlgebra
using OrdinaryDiffEq   
using Plots

# These helpers are not part of the LyapunovSpectra. This is only to guarantee a "safe" energy window for the initial conditions, so we can reproduce the paper's results more closely.
# ── Helper: compute Hamiltonian energy ───────────────────────────────────────
function hamiltonian_energy(u)
    x, y, z    = u[1], u[2], u[3]
    px, py, pz = u[4], u[5], u[6]
    return (px^2 + py^2 + pz^2)/2 +
           x^2*y^2 + y^2*z^2 + z^2*x^2 +
           (x^4 + y^4 + z^4)/32
end

# ── Helper: generate initial conditions in a safe energy window ───────────────
function generate_initial_conditions(n, base, rng)
    x0_list = Vector{Vector{Float64}}()
    E_base  = hamiltonian_energy(base)
    # Accept points within ±30% of the base energy
    E_lo    = 0.7 * E_base
    E_hi    = 1.3 * E_base

    while length(x0_list) < n
        x0 = 0.05 .* randn(rng, 6) .+ base
        E  = hamiltonian_energy(x0)
        if E_lo < E < E_hi
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

# ── Warm-up: push initial condition onto the attractor ────────────────────────
# We integrate the Hamiltonian system (not the augmented one) for a short
# time from an arbitrary starting point. The final state is on the attractor
# and is used as the true initial condition for the main run.
# This eliminates the transient visible at small t in Figure 2.
#
# We reuse build_ode! with k=1 to evolve only the trajectory (cheaply).
x0_arbitrary = [0.5, 0.3, 0.7, 0.3, 0.2, 0.4]
T_warmup     = 200.0   

println("\nWarm-up integration:")
@printf "  Starting from x0 = [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n" x0_arbitrary...
@printf "  Energy: H = %.4f\n" hamiltonian_energy(x0_arbitrary)
println("  Integrating for T_warmup = $T_warmup ...")

prob_warmup  = LyapunovProblem(prob.v, prob.J, prob.d, 1, prob.β)
u0_warmup    = make_initial_state(prob_warmup, x0_arbitrary)
ode_warmup!  = build_ode!(prob_warmup)
warmup_sol   = solve(
    ODEProblem(ode_warmup!, u0_warmup, (0.0, T_warmup), nothing),
    Tsit5(); reltol=1e-8, abstol=1e-8, save_everystep=false)

# Extract just the trajectory part (first d=6 components of the augmented state)
x0_single = warmup_sol.u[end][1:prob.d]

@printf "  Warm-up complete → x0 = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n" x0_single...
@printf "  Energy after warm-up: H = %.4f\n" hamiltonian_energy(x0_single)

# ── Single run + convergence plot (reproduces Figure 2) ──────────────────────
# We solve the full augmented system with save_everystep=true so we can
# plot λₘ(t) = Λₘ(t)/t at every time step — matching Figure 2 of the paper.
# The paper only shows the three positive exponents (λ₁, λ₂, λ₃) since the
# negative ones are their exact mirror images by symplectic symmetry.

println("\nSingle run (reproduces Figure 2 of the paper):")
println("  Integrating for T = $T (saving all steps for the plot)...")

u0   = make_initial_state(prob, x0_single)
ode! = build_ode!(prob)
sol  = solve(
    ODEProblem(ode!, u0, (0.0, T), nothing),
    Tsit5();
    reltol         = 1e-6,
    abstol         = 1e-6,
    maxiters       = 1_000_000,
    save_everystep = true)

# Extract time vector and compute λₘ(t) = Λₘ(t)/t at every saved step.
# Skip t=0 to avoid division by zero.
d  = prob.d
k  = prob.k
ts = sol.t[2:end]

# λ_series: matrix of size (n_timepoints × k)
# Each row is the full spectrum at one time point.
λ_series = hcat([sol.u[i][d + d*k + 1 : d + d*k + k] ./ sol.t[i]
                 for i in 2:length(sol.t)]...)'

# Print final values and symplectic check
λ_single = λ_series[end, :]

println("\n  Full spectrum:")
println("  ┌─────┬────────────┬─────────────────────────────┐")
println("  │  k  │    λₖ      │  symplectic check           │")
println("  ├─────┼────────────┼─────────────────────────────┤")
for ki in 1:6
    pair_k   = 7 - ki
    pair_sum = λ_single[ki] + λ_single[pair_k]
    @printf "  │  %d  │  %+.4f  │  λ%d + λ%d = %+.6f      │\n" ki λ_single[ki] ki pair_k pair_sum
end
println("  └─────┴────────────┴─────────────────────────────┘")

println("\n  Positive exponents only (paper Table 2, T=10000):")
@printf "  λ₁ = %+.4f  (paper: +0.2374)\n" λ_single[1]
@printf "  λ₂ = %+.4f  (paper: +0.1184)\n" λ_single[2]
@printf "  λ₃ = %+.4f  (paper: +0.000390)\n" λ_single[3]

# ── Figure 2: convergence of positive finite-time Lyapunov exponents ──────────
# Three stacked panels showing λ₁(t), λ₂(t), λ₃(t) converging over time.
# The paper only shows positive exponents since λ₄ = −λ₃, λ₅ = −λ₂, λ₆ = −λ₁
# by symplectic symmetry — no new information in the negative ones.
println("\n  Generating Figure 2 (convergence plot)...")

colours = [:steelblue, :darkorange, :crimson]

p1 = plot(ts, λ_series[:, 1];
          color     = colours[1],
          linewidth = 1.5,
          ylabel    = "λ₁(t)",
          label     = "",
          title     = "Quartic Hamiltonian — finite-time Lyapunov exponents  (Figure 2)",
          titlefont = font(10))

hline!(p1, [0.2374]; color=:black, linestyle=:dash, linewidth=1, label="paper λ₁")

p2 = plot(ts, λ_series[:, 2];
          color     = colours[2],
          linewidth = 1.5,
          ylabel    = "λ₂(t)",
          label     = "")

hline!(p2, [0.1184]; color=:black, linestyle=:dash, linewidth=1, label="paper λ₂")

p3 = plot(ts, λ_series[:, 3];
          color     = colours[3],
          linewidth = 1.5,
          xlabel    = "t",
          ylabel    = "λ₃(t)",
          label     = "")

hline!(p3, [0.000390]; color=:black, linestyle=:dash, linewidth=1, label="paper λ₃")

fig2 = plot(p1, p2, p3;
            layout = (3, 1),
            size   = (700, 600),
            margin = 5Plots.mm)

savefig(fig2, "hamiltonian_figure2.png")
println("  Saved: hamiltonian_figure2.png")

# ── n_runs average (reproduces Table 2) ──────────────────────────────────────
println("\n$(n_runs)-run average (reproduces Table 2 of the paper):")

rng_multi = Random.MersenneTwister(0)

# Generate initial conditions as small perturbations around the warm-up point.
# We use x0_single (already on the attractor) as the base, so all runs
# start in the same energy region and the energy filter is tight.
println("  Generating $n_runs initial conditions around warm-up point ...")
x0_list  = generate_initial_conditions(n_runs, x0_single, rng_multi)
energies = hamiltonian_energy.(x0_list)
@printf "  Energy range: [%.3f, %.3f]\n" minimum(energies) maximum(energies)

println("  Running $n_runs integrations of length T=$T ...")
println("  (each run is a 6D system — this will take a few minutes)\n")

elapsed = @elapsed begin
    λ_mean, λ_std = lyapunov_exponents_mean(prob, x0_list, T)
end

# Full spectrum
println("  Full spectrum results:")
println("  ┌─────┬────────────┬────────────┐")
println("  │  k  │    λₖ      │  rms dev   │")
println("  ├─────┼────────────┼────────────┤")
for ki in 1:6
    @printf "  │  %d  │  %+.4f  │  %.2e  │\n" ki λ_mean[ki] λ_std[ki]
end
println("  └─────┴────────────┴────────────┘")

# Positive exponents vs paper
println("\n  Positive exponents vs paper (Table 2):")
println("  ┌─────┬────────────┬────────────┬────────────┐")
println("  │  k  │    λₖ      │  rms dev   │  paper λₖ  │")
println("  ├─────┼────────────┼────────────┼────────────┤")
@printf "  │  1  │  %+.4f  │  %.2e  │  +0.2374   │\n" λ_mean[1] λ_std[1]
@printf "  │  2  │  %+.4f  │  %.2e  │  +0.1184   │\n" λ_mean[2] λ_std[2]
@printf "  │  3  │  %+.4f  │  %.2e  │  +0.000390 │\n" λ_mean[3] λ_std[3]
println("  └─────┴────────────┴────────────┴────────────┘")

# Symplectic check on averages
println("\n  Symplectic pair sums (should all be ~0):")
for ki in 1:3
    pair_k   = 7 - ki
    pair_sum = λ_mean[ki] + λ_mean[pair_k]
    @printf "  λ%d + λ%d = %+.6f\n" ki pair_k pair_sum
end

@printf "\n  Total wall time: %.1f seconds\n" elapsed