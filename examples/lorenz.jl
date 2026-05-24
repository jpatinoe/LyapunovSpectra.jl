# examples/lorenz.jl
#
# Reproduces the Lorenz results from Christiansen & Rugh (1997):
#   - Figure 1: finite-time Lyapunov exponents from a single run
#   - Table  1: Mean and RMS deviation over 1000 runs
#
# Run this script from the terminal with:
#   julia --project=. examples/lorenz.jl
# ─────────────────────────────────────────────────────────────────────────────
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using LyapunovSpectra
using Random
using Statistics
using Printf
using OrdinaryDiffEq   
using Plots

println("="^60)
println("  Lorenz System — as in Christiansen & Rugh (1997)")
println("="^60)

# ── Parameters ────────────────────────────────────────────────────────────────
T      = 1000.0
n_runs = 1000

# Build the problem struct (d=3, k=3, β=20). Already defined in src/systems.jl
prob = lorenz_system()

# ── Single run + convergence plot (reproduces Figure 1) ──────────────────────
# For the plot we need λₘ(t) = Λₘ(t)/t at every time step, not just at T.
# So we run the ODE solver with save_everystep=true for this one case,
# accessing the augmented system directly.

# ── Warm-up: push the initial condition onto the attractor ────────────────────
# Starting from an arbitrary point [1,0,0], we first integrate the original
# Lorenz system (not the augmented one) for a short time to land close to the
# attractor. The final state is then used as the true initial condition.
# This eliminates the transient and makes the finite-time exponents more
# representative of the attractor's actual Lyapunov spectrum.

x0_arbitrary = [1.0, 0.0, 0.0]
T_warmup     = 50.0    # short enough to be cheap, long enough to reach attractor

# Solve just the original 3D Lorenz system (not augmented) for the warm-up.
# Build a k=1 problem just for the warm-up — we only need the trajectory
prob_warmup = LyapunovProblem(prob.v, prob.J, prob.d, 1, prob.β)
u0_warmup   = make_initial_state(prob_warmup, x0_arbitrary)
ode_warmup! = build_ode!(prob_warmup)

warmup_sol = solve(
    ODEProblem(ode_warmup!, u0_warmup, (0.0, T_warmup), nothing),
    Tsit5(); reltol=1e-8, abstol=1e-8, save_everystep=false)

# Extract just the trajectory part (first d components)
x0_single = warmup_sol.u[end][1:prob.d]

println("\nSingle run (reproduces Figure 1 of the paper):")
@printf "  Warm-up from [1,0,0] for T=%.0f → x0 = [%.3f, %.3f, %.3f]\n" T_warmup x0_single[1] x0_single[2] x0_single[3]
println("  Integrating for T = $T (saving all steps for the plot)...")

# Build the augmented ODE and initial state — same as lyapunov_exponents()
# does internally, but here we call solve() with save_everystep=true so we
# get the full time series of Λ(t), not just the final value.
u0   = make_initial_state(prob, x0_single)
ode! = build_ode!(prob)
sol  = solve(ODEProblem(ode!, u0, (0.0, T), nothing),
             Tsit5();
             reltol         = 1e-6,
             abstol         = 1e-6,
             maxiters       = 1_000_000,
             save_everystep = true)   # ← save every step for the time series

# Extract time vector and compute λₘ(t) = Λₘ(t)/t at every saved step.
# Skip t=0 to avoid division by zero.
d = prob.d
k = prob.k

# sol.t is the vector of saved time points
# sol.u is the vector of saved state vectors
ts = sol.t[2:end]   # skip t=0

# For each saved time point, extract Λ(t) and divide by t
# λ_series is a matrix: rows = time points, columns = exponents
λ_series = hcat([sol.u[i][d + d*k + 1 : d + d*k + k] ./ sol.t[i]
                 for i in 2:length(sol.t)]...)'

# Print final values
λ_single = λ_series[end, :]
@printf "  λ₁ = %+.4f  (paper: +0.9057)\n"  λ_single[1]
@printf "  λ₂ = %+.4f  (paper: ~0.0000)\n"  λ_single[2]
@printf "  λ₃ = %+.4f  (paper: -14.5724)\n" λ_single[3]
@printf "  Sum = %.4f  (should equal -σ-1-b = -%.4f)\n" sum(λ_single) (10+1+8/3)

# ── Figure 1: convergence of finite-time Lyapunov exponents ──────────────────
println("\n  Generating Figure 1 (convergence plot)...")

colours = [:steelblue, :darkorange, :crimson]
labels  = ["λ₁(t)" "λ₂(t)" "λ₃(t)"]

# Three stacked panels, one per exponent
p1 = plot(ts, λ_series[:, 1];
          color     = colours[1],
          linewidth = 1.5,
          xlabel    = "",
          ylabel    = "λ₁(t)",
          label     = "",
          title     = "Lorenz system — finite-time Lyapunov exponents  (Figure 1)",
          titlefont = font(10))

hline!(p1, [0.9057]; color=:black, linestyle=:dash, linewidth=1, label="paper λ₁")

p2 = plot(ts, λ_series[:, 2];
          color     = colours[2],
          linewidth = 1.5,
          xlabel    = "",
          ylabel    = "λ₂(t)",
          label     = "",
          ylims     = (-0.15, 0.15))

hline!(p2, [0.0]; color=:black, linestyle=:dash, linewidth=1, label="paper λ₂ ≈ 0")

p3 = plot(ts, λ_series[:, 3];
          color     = colours[3],
          linewidth = 1.5,
          xlabel    = "t",
          ylabel    = "λ₃(t)",
          label     = "")

hline!(p3, [-14.5724]; color=:black, linestyle=:dash, linewidth=1, label="paper λ₃")

fig1 = plot(p1, p2, p3;
            layout  = (3, 1),
            size    = (700, 600),
            margin  = 5Plots.mm)

savefig(fig1, "lorenz_figure1.png")
println("  Saved: lorenz_figure1.png")

# ── 1000 runs (reproduces Table 1) ────────────────────────────────────────────
println("\n1000-run average (reproduces Table 1 of the paper):")
println("  Generating $n_runs random initial conditions ...")

rng_multi = Random.MersenneTwister(0)

x0_list = [randn(rng_multi, 3) .* 0.1 .+ x0_single
           for _ in 1:n_runs]

println("  Running $n_runs integrations of length T=$T ...")
println("  (this will take a minute — 1000 × T=1000 is a lot of work)\n")

elapsed = @elapsed begin
    λ_mean, λ_std = lyapunov_exponents_mean(prob, x0_list, T)
end

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