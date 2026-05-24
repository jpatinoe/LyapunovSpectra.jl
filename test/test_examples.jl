# test/test_examples.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE TEST SUITE — self-contained entry point.
#
# For each example system this file:
#   1. Computes the Lyapunov exponents
#   2. Calls the generic algorithm tests from test_algorithm.jl
#   3. Adds example-specific assertions (known values from the paper)
#
# Run with:
#   julia --project=. test/test_examples.jl
#
# To test your own system, follow the same pattern:
#   1. Define prob, x0, T
#   2. Compute λ = lyapunov_exponents(prob, x0, T)
#   3. Call test_dissipative(prob, λ, x0, T)  or  test_hamiltonian(prob, λ, x0, T)
#   4. Add any system-specific @test assertions below
# ─────────────────────────────────────────────────────────────────────────────

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using LyapunovSpectra
using Test
using Printf
using LinearAlgebra
using OrdinaryDiffEq

# Load the generic algorithm test functions
include("test_algorithm.jl")

println("="^60)
println("  LyapunovSpectra.jl — Test Suite")
println("="^60)


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 1: LORENZ SYSTEM
#  Dissipative system — use test_dissipative()
#  Reference: Christiansen & Rugh (1997), Table 1
# ══════════════════════════════════════════════════════════════════════════════

println("\n[ Lorenz system ]")

prob_lorenz = lorenz_system()
x0_lorenz   = [1.0, 0.0, 0.0]
T_lorenz    = 2000.0

λ_lorenz = lyapunov_exponents(prob_lorenz, x0_lorenz, T_lorenz)
@printf "  λ = [%+.4f, %+.4f, %+.4f]\n\n" λ_lorenz[1] λ_lorenz[2] λ_lorenz[3]

# ── Generic algorithm tests for any dissipative system ────────────────────────
test_dissipative(prob_lorenz, λ_lorenz, x0_lorenz, T_lorenz)

# ── Lorenz-specific tests ─────────────────────────────────────────────────────

# E1: system is chaotic — largest exponent must be positive
@testset "E1: Lorenz λ₁ > 0 (chaotic attractor)" begin
    println("  E1: Lorenz λ₁ > 0")
    @printf "      λ₁ = %+.4f  (should be > 0)\n" λ_lorenz[1]
    @test λ_lorenz[1] > 0.0
end

# E2: sum = tr(J) = −σ−1−b = −13.6̄ (exact analytical result for Lorenz)
# This is specific to σ=10, r=28, b=8/3 — not a generic property.
@testset "E2: Lorenz Σλ = −σ−1−b = −13.6667 (Liouville)" begin
    println("  E2: Lorenz sum of exponents")
    expected = -(10.0 + 1.0 + 8.0/3.0)
    actual   = sum(λ_lorenz)
    @printf "      Σλ = %.6f  (expected %.6f)\n" actual expected
    @test actual ≈ expected atol=1e-4
end

# E3: λ₁ close to paper value (Table 1: 0.9057)
# Loose tolerance — finite-time estimates vary with initial condition and T.
@testset "E3: Lorenz λ₁ ≈ 0.9057 (paper Table 1, tolerance ±0.05)" begin
    println("  E3: Lorenz λ₁ vs paper")
    @printf "      λ₁ = %+.4f  (paper: +0.9057)\n" λ_lorenz[1]
    @test abs(λ_lorenz[1] - 0.9057) < 0.05
end


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 2: QUARTIC HAMILTONIAN
#  Hamiltonian system — use test_hamiltonian()
#  Reference: Christiansen & Rugh (1997), Table 2
# ══════════════════════════════════════════════════════════════════════════════

println("\n[ Quartic Hamiltonian system ]")

prob_ham = hamiltonian_system()
x0_ham   = [0.5, 0.3, 0.7, 0.3, 0.2, 0.4]
T_ham    = 5000.0

λ_ham = lyapunov_exponents(prob_ham, x0_ham, T_ham)
@printf "  λ = [%+.4f, %+.4f, %+.4f, %+.4f, %+.4f, %+.4f]\n\n" λ_ham...

# ── Generic algorithm tests for any Hamiltonian system ────────────────────────
# This calls test_dissipative internally, then adds symplectic + sum checks.
test_hamiltonian(prob_ham, λ_ham, x0_ham, T_ham)

# ── Hamiltonian-specific tests ────────────────────────────────────────────────

# E4: system is chaotic at this energy level
@testset "E4: Hamiltonian λ₁ > 0 (chaotic at this energy)" begin
    println("  E4: Hamiltonian λ₁ > 0")
    @printf "      λ₁ = %+.4f  (should be > 0)\n" λ_ham[1]
    @test λ_ham[1] > 0.0
end

# E5: three positive and three negative exponents
# This follows from symplectic pairing but is worth checking explicitly.
@testset "E5: Hamiltonian sign structure (3 positive, 3 negative)" begin
    println("  E5: Hamiltonian sign structure")
    @printf "      positive: λ₁=%+.4f  λ₂=%+.4f  λ₃=%+.4f\n" λ_ham[1] λ_ham[2] λ_ham[3]
    @printf "      negative: λ₄=%+.4f  λ₅=%+.4f  λ₆=%+.4f\n" λ_ham[4] λ_ham[5] λ_ham[6]
    @test all(λ_ham[1:3] .> 0.0)
    @test all(λ_ham[4:6] .< 0.0)
end

# E6: positive exponents in expected range compared to paper (Table 2)
# Generous tolerance — our initial condition and energy differ from the
# paper's unspecified choice, so exact agreement is not expected.
@testset "E6: Hamiltonian positive exponents in expected range (paper Table 2)" begin
    println("  E6: Hamiltonian λ₁, λ₂ vs paper")
    @printf "      λ₁ = %+.4f  (paper: ~0.24, tolerance ±0.15)\n" λ_ham[1]
    @printf "      λ₂ = %+.4f  (paper: ~0.12, tolerance ±0.10)\n" λ_ham[2]
    @test abs(λ_ham[1] - 0.24) < 0.15
    @test abs(λ_ham[2] - 0.12) < 0.10
end

println("\n" * "="^60)
println("  All tests complete.")
println("="^60)
