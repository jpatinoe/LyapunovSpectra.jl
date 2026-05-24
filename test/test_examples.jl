# test/test_examples.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE TEST SUITE — self-contained entry point.
#
# For each example system this file:
#   1. Computes λ and saves the LyapunovState (return_state=true)
#   2. Calls the appropriate generic test function from test_algorithm.jl
#   3. Adds system-specific assertions against known values from the paper
#
# Run with:
#   julia --project=. test/test_examples.jl
#
# To test your own system, follow the same pattern:
#   1. Compute: λ, state = lyapunov_exponents(prob, x0, T; return_state=true)
#   2. Call:    test_dissipative(prob, λ, state)
#              or test_hamiltonian(prob, λ, state)
#   3. Add any system-specific @test assertions
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
#  EXAMPLE 1: LORENZ SYSTEM (dissipative)
#  Reference: Christiansen & Rugh (1997), Table 1
# ══════════════════════════════════════════════════════════════════════════════

println("\n[ Lorenz system ]")

prob_lorenz = lorenz_system()
x0_lorenz   = [1.0, 0.0, 0.0]
T_lorenz    = 2000.0

# Compute exponents and save state in one call
λ_lorenz, state_lorenz = lyapunov_exponents(prob_lorenz, x0_lorenz, T_lorenz;
                                             return_state = true)

@printf "  λ = [%+.4f, %+.4f, %+.4f]\n\n" λ_lorenz[1] λ_lorenz[2] λ_lorenz[3]

# ── Generic algorithm tests ───────────────────────────────────────────────────
# D1: descending order     — uses λ only
# D2: one near zero        — uses λ only
# D3: frame orthonormality — uses state (no re-integration)
# D4: k=1 consistency      — runs one extra integration from state.u[1:d]
test_dissipative(prob_lorenz, λ_lorenz, state_lorenz)

# ── Lorenz-specific tests ─────────────────────────────────────────────────────

@testset "E1: Lorenz λ₁ > 0 (chaotic attractor)" begin
    println("  E1: Lorenz λ₁ > 0")
    @printf "      λ₁ = %+.4f  (should be > 0)\n" λ_lorenz[1]
    @test λ_lorenz[1] > 0.0
end

@testset "E2: Lorenz Σλ = −σ−1−b = −13.6667 (Liouville)" begin
    println("  E2: Lorenz sum of exponents")
    expected = -(10.0 + 1.0 + 8.0/3.0)
    actual   = sum(λ_lorenz)
    @printf "      Σλ = %.6f  (expected %.6f)\n" actual expected
    @test actual ≈ expected atol=1e-4
end

@testset "E3: Lorenz λ₁ ≈ 0.9057 (paper Table 1, tolerance ±0.05)" begin
    println("  E3: Lorenz λ₁ vs paper")
    @printf "      λ₁ = %+.4f  (paper: +0.9057)\n" λ_lorenz[1]
    @test abs(λ_lorenz[1] - 0.9057) < 0.05
end

# ── Restart test ──────────────────────────────────────────────────────────────
# Verify that restarting from state_lorenz and integrating for another T
# gives the same result as integrating for 2T from scratch.
# This validates the LyapunovState restart mechanism.
@testset "E4: Lorenz restart consistency" begin
    println("  E4: Restart gives same result as single long run")

    T_extra = 1000.0

    # Option A: restart from saved state
    λ_restart, _ = lyapunov_exponents(prob_lorenz, state_lorenz, T_extra;
                                       return_state = true)

    # Option B: single run for T_lorenz + T_extra
    λ_long = lyapunov_exponents(prob_lorenz, x0_lorenz, T_lorenz + T_extra)

    @printf "      λ₁ (restart) = %+.4f\n" λ_restart[1]
    @printf "      λ₁ (long)    = %+.4f  (tolerance: ±0.05)\n" λ_long[1]
    @test λ_restart[1] ≈ λ_long[1] atol=0.05
end


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 2: QUARTIC HAMILTONIAN
#  Reference: Christiansen & Rugh (1997), Table 2
# ══════════════════════════════════════════════════════════════════════════════

println("\n[ Quartic Hamiltonian system ]")

prob_ham = hamiltonian_system()
x0_ham   = [0.5, 0.3, 0.7, 0.3, 0.2, 0.4]
T_ham    = 5000.0

λ_ham, state_ham = lyapunov_exponents(prob_ham, x0_ham, T_ham;
                                       return_state = true)

@printf "  λ = [%+.4f, %+.4f, %+.4f, %+.4f, %+.4f, %+.4f]\n\n" λ_ham...

# ── Generic algorithm tests ───────────────────────────────────────────────────
# Calls test_dissipative internally, then adds H1 (symplectic) and H2 (sum=0)
test_hamiltonian(prob_ham, λ_ham, state_ham)

# ── Hamiltonian-specific tests ────────────────────────────────────────────────

@testset "E5: Hamiltonian λ₁ > 0 (chaotic at this energy)" begin
    println("  E5: Hamiltonian λ₁ > 0")
    @printf "      λ₁ = %+.4f  (should be > 0)\n" λ_ham[1]
    @test λ_ham[1] > 0.0
end

@testset "E6: Hamiltonian sign structure (3 positive, 3 negative)" begin
    println("  E6: Hamiltonian sign structure")
    @printf "      positive: λ₁=%+.4f  λ₂=%+.4f  λ₃=%+.4f\n" λ_ham[1] λ_ham[2] λ_ham[3]
    @printf "      negative: λ₄=%+.4f  λ₅=%+.4f  λ₆=%+.4f\n" λ_ham[4] λ_ham[5] λ_ham[6]
    @test all(λ_ham[1:3] .> 0.0)
    @test all(λ_ham[4:6] .< 0.0)
end

@testset "E7: Hamiltonian positive exponents in expected range (paper Table 2)" begin
    println("  E7: Hamiltonian λ₁, λ₂ vs paper")
    @printf "      λ₁ = %+.4f  (paper: ~0.24, tolerance ±0.15)\n" λ_ham[1]
    @printf "      λ₂ = %+.4f  (paper: ~0.12, tolerance ±0.10)\n" λ_ham[2]
    @test abs(λ_ham[1] - 0.24) < 0.15
    @test abs(λ_ham[2] - 0.12) < 0.10
end

println("\n" * "="^60)
println("  All tests complete.")
println("="^60)
