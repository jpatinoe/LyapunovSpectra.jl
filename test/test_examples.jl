# test/test_examples.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE-SPECIFIC TESTS
#
# These tests check known numerical results for the two example systems
# from the paper. They verify that our implementation reproduces the
# qualitative behaviour reported in Christiansen & Rugh (1997).
#
# Unlike test_algorithm.jl, these tests are tied to specific systems and
# specific parameter choices. If you modify the example systems (different
# parameters, different β) these tests may need updating.
# ─────────────────────────────────────────────────────────────────────────────

using Test
using LyapunovSpectra

println("─"^60)
println("  Example-specific tests")
println("─"^60)

# ── Lorenz system tests ───────────────────────────────────────────────────────
# Known facts about the Lorenz attractor with σ=10, r=28, b=8/3:
#   λ₁ ≈ +0.906  (positive → chaotic)
#   λ₂ ≈  0      (neutral → along the flow)
#   λ₃ ≈ −14.57  (strongly negative → thin attractor)

prob_lorenz = lorenz_system()
x0_lorenz   = [1.0, 0.0, 0.0]
T_lorenz    = 2000.0
λ_lorenz    = lyapunov_exponents(prob_lorenz, x0_lorenz, T_lorenz)

println("\nLorenz system (σ=10, r=28, b=8/3):")
@printf "    λ₁ = %+.4f  λ₂ = %+.4f  λ₃ = %+.4f\n" λ_lorenz[1] λ_lorenz[2] λ_lorenz[3]

# E1: largest exponent is positive — the Lorenz system is chaotic
@testset "E1: Lorenz λ₁ > 0 (system is chaotic)" begin
    println("\nE1: Lorenz λ₁ > 0")
    @printf "    λ₁ = %+.4f  (should be > 0)\n" λ_lorenz[1]
    @test λ_lorenz[1] > 0.0
end

# E2: second exponent near zero — neutral direction along the flow.
# For any autonomous system, one exponent must be zero (or near zero
# for finite T) because the flow direction neither grows nor shrinks.
@testset "E2: Lorenz λ₂ ≈ 0 (neutral direction along flow)" begin
    println("\nE2: Lorenz λ₂ ≈ 0")
    @printf "    λ₂ = %+.6f  (should be ~0, threshold 0.01)\n" λ_lorenz[2]
    @test abs(λ_lorenz[2]) < 0.01
end

# E3: largest exponent is in the right ballpark compared to the paper.
# The paper reports λ₁ ≈ 0.9057 (Table 1). We use a loose tolerance
# because finite-time estimates vary with initial condition and T.
@testset "E3: Lorenz λ₁ close to paper value (0.9057 ± 0.05)" begin
    println("\nE3: Lorenz λ₁ close to paper value")
    @printf "    λ₁ = %+.4f  (paper: +0.9057, tolerance ±0.05)\n" λ_lorenz[1]
    @test abs(λ_lorenz[1] - 0.9057) < 0.05
end

# ── Hamiltonian system tests ──────────────────────────────────────────────────
# Known facts about the quartic Hamiltonian (eq. 10) with our initial condition:
#   λ₁ > 0   (system is chaotic at this energy)
#   λ₁ > λ₂ > λ₃ > 0  (three positive exponents, rest negative by pairing)

prob_ham = hamiltonian_system()
x0_ham   = [0.5, 0.3, 0.7, 0.3, 0.2, 0.4]
T_ham    = 5000.0
λ_ham    = lyapunov_exponents(prob_ham, x0_ham, T_ham)

println("\nQuartic Hamiltonian system:")
@printf "    λ₁=%+.4f  λ₂=%+.4f  λ₃=%+.4f\n" λ_ham[1] λ_ham[2] λ_ham[3]
@printf "    λ₄=%+.4f  λ₅=%+.4f  λ₆=%+.4f\n" λ_ham[4] λ_ham[5] λ_ham[6]

# E4: system is chaotic at this energy level
@testset "E4: Hamiltonian λ₁ > 0 (system is chaotic)" begin
    println("\nE4: Hamiltonian λ₁ > 0")
    @printf "    λ₁ = %+.4f  (should be > 0)\n" λ_ham[1]
    @test λ_ham[1] > 0.0
end

# E5: three positive and three negative exponents (by symplectic pairing)
@testset "E5: Hamiltonian has three positive and three negative exponents" begin
    println("\nE5: Hamiltonian exponent sign structure")
    @printf "    positive: λ₁=%+.4f  λ₂=%+.4f  λ₃=%+.4f\n" λ_ham[1] λ_ham[2] λ_ham[3]
    @printf "    negative: λ₄=%+.4f  λ₅=%+.4f  λ₆=%+.4f\n" λ_ham[4] λ_ham[5] λ_ham[6]
    @test λ_ham[1] > 0.0
    @test λ_ham[2] > 0.0
    @test λ_ham[3] > 0.0
    @test λ_ham[4] < 0.0
    @test λ_ham[5] < 0.0
    @test λ_ham[6] < 0.0
end

# E6: positive exponents close to paper values (Table 2).
# The paper reports λ₁≈0.2374, λ₂≈0.1184 for their specific initial
# conditions. We use a generous tolerance (±0.15) because our initial
# condition and energy differ from the paper's unspecified choice.
@testset "E6: Hamiltonian positive exponents in expected range" begin
    println("\nE6: Hamiltonian positive exponents in expected range")
    @printf "    λ₁ = %+.4f  (paper: ~0.24, tolerance ±0.15)\n" λ_ham[1]
    @printf "    λ₂ = %+.4f  (paper: ~0.12, tolerance ±0.10)\n" λ_ham[2]
    @test abs(λ_ham[1] - 0.24) < 0.15
    @test abs(λ_ham[2] - 0.12) < 0.10
end

println()
