# test/runtests.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE ENTRY POINT
#
# Runs two separate test files:
#
#   test_algorithm.jl  — generic tests that must pass for ANY correct
#                        implementation of the CGS Lyapunov method.
#                        Tests mathematical properties of the algorithm
#                        itself: ordering, partial spectrum, frame
#                        orthonormality, trace identity, symplectic pairing.
#
#   test_examples.jl   — example-specific tests tied to the two systems
#                        from Christiansen & Rugh (1997): the Lorenz
#                        attractor and the quartic Hamiltonian. Tests
#                        known qualitative facts (chaos, neutral direction,
#                        sign structure) and rough agreement with the
#                        paper's Table 1 and Table 2.
#
# Run with:
#   julia --project=. test/runtests.jl
# ─────────────────────────────────────────────────────────────────────────────

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using LyapunovSpectra
using Test
using Printf
using LinearAlgebra
using OrdinaryDiffEq

println("="^60)
println("  LyapunovSpectra.jl — Test Suite")
println("="^60)

# ── Run generic algorithm tests ───────────────────────────────────────────────
println("\n[ Part 1: Generic algorithm tests ]")
include("test_algorithm.jl")

# ── Run example-specific tests ────────────────────────────────────────────────
println("[ Part 2: Example-specific tests ]")
include("test_examples.jl")

println("="^60)
println("  All tests complete.")
println("="^60)
