# test/test_algorithm.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# GENERIC ALGORITHM TESTS
#
# These tests check mathematical properties that must hold for ANY correct
# implementation of the continuous Gram-Schmidt Lyapunov method, regardless
# of which dynamical system is used. They test the algorithm itself, not
# the specific numerical values of any particular example.
#
# If any of these fail, something is wrong with the core algorithm in
# algorithm.jl or solver.jl — not just with the example systems.
# ─────────────────────────────────────────────────────────────────────────────

using Test
using LinearAlgebra
using OrdinaryDiffEq
using LyapunovSpectra

println("─"^60)
println("  Generic algorithm tests")
println("─"^60)

# We use the Lorenz system as the test vehicle throughout, but the properties
# being checked are not Lorenz-specific — they follow from the mathematics
# of the method and would hold for any well-behaved dynamical system.
prob = lorenz_system()
x0   = [1.0, 0.0, 0.0]
T    = 2000.0

# Compute once and reuse across tests to avoid redundant integrations.
λ = lyapunov_exponents(prob, x0, T)

# ── Test A1: Exponents in descending order ────────────────────────────────────
# The Gram-Schmidt ordering guarantees λ₁ ≥ λ₂ ≥ … ≥ λₖ automatically.
# This must hold for any system and any initial condition — it is a
# structural property of the algorithm, not a numerical coincidence.
@testset "A1: Exponents in descending order" begin
    println("\nA1: Exponents in descending order")
    for m in 1:length(λ)-1
        @printf "    λ%d = %+.4f ≥ λ%d = %+.4f\n" m λ[m] (m+1) λ[m+1]
        @test λ[m] ≥ λ[m+1]
    end
end

# ── Test A2: Partial spectrum consistent with full spectrum ───────────────────
# Computing only k=1 exponent should give the same λ₁ as the full spectrum.
# This tests that reducing k does not corrupt the largest exponent —
# a property that follows from the Gram-Schmidt ordering.
# We use a loose tolerance (0.05) because the two runs use different
# integration paths and accumulate different floating-point rounding.
@testset "A2: Partial spectrum (k=1) consistent with full spectrum" begin
    println("\nA2: Partial spectrum k=1 consistent with full spectrum")

    prob_partial = LyapunovProblem(prob.v, prob.J, prob.d, 1, prob.β)
    λ_partial    = lyapunov_exponents(prob_partial, x0, T)

    @printf "    λ₁ (k=%d) = %+.4f\n" prob.k   λ[1]
    @printf "    λ₁ (k=1)  = %+.4f\n"          λ_partial[1]
    @test λ[1] ≈ λ_partial[1] atol=0.05
end

# ── Test A3: Frame stays orthonormal ─────────────────────────────────────────
# The β stabilisation term in equation (6) is designed to keep the frame
# E = {e₁,…,eₖ} orthonormal at all times. We verify this by checking
# ‖E'E − I‖ at the end of the integration.
# This tests the stabilisation mechanism directly, for any β > 0.
@testset "A3: Frame orthonormality preserved by β stabilisation" begin
    println("\nA3: Frame orthonormality ‖E'E − I‖ < 1e-6")

    d = prob.d
    k = prob.k

    u0       = make_initial_state(prob, x0)
    ode!     = build_ode!(prob)
    ode_prob = ODEProblem(ode!, u0, (0.0, T), nothing)
    sol      = solve(ode_prob, Tsit5();
                     reltol=1e-8, abstol=1e-8,
                     maxiters=1_000_000,
                     save_everystep=false)

    E_final = reshape(sol.u[end][d+1 : d+d*k], d, k)
    err     = norm(E_final' * E_final - I)

    @printf "    ‖E'E − I‖ = %.2e  (threshold: 1e-6)\n" err
    @test err < 1e-6
end

# ── Test A4: Sum of exponents equals trace of Jacobian ────────────────────────
# For any autonomous ODE, the sum of all Lyapunov exponents equals the
# time-average of tr(J(x(t))). For the Lorenz system, tr(J) = −σ−1−b
# is CONSTANT along every trajectory, so the average is exact analytically.
# This is Liouville's theorem applied to the flow.
# We use Lorenz here because its Jacobian has constant trace — this makes
# the test exact and sharp (atol=1e-4). For systems with non-constant
# tr(J), the equality still holds but requires longer T to converge.
@testset "A4: Sum of exponents = time-average of tr(J)" begin
    println("\nA4: Sum of exponents = tr(J) = −σ−1−b")

    expected_sum = -(10.0 + 1.0 + 8.0/3.0)   # exact: tr(J) = −σ−1−b
    actual_sum   = sum(λ)

    @printf "    Σλₘ = %.6f  (expected %.6f)\n" actual_sum expected_sum
    @test actual_sum ≈ expected_sum atol=1e-4
end

# ── Test A5: Symplectic pairing for a Hamiltonian system ──────────────────────
# For any Hamiltonian system, symplectic structure guarantees that Lyapunov
# exponents come in conjugate pairs: λₖ + λ_{d+1−k} = 0 for all k.
# This is a fundamental property of Hamiltonian dynamics (not of our
# specific example) and provides a strong validation of the frame dynamics.
@testset "A5: Symplectic pairing for Hamiltonian system" begin
    println("\nA5: Symplectic pairs λₖ + λ_{7-k} ≈ 0")

    prob_ham = hamiltonian_system()
    x0_ham   = [0.5, 0.3, 0.7, 0.3, 0.2, 0.4]
    λ_ham    = lyapunov_exponents(prob_ham, x0_ham, 5000.0)

    for k in 1:3
        pair_sum = λ_ham[k] + λ_ham[7-k]
        @printf "    λ%d + λ%d = %+.6f  (threshold: 1e-3)\n" k (7-k) pair_sum
        @test abs(pair_sum) < 1e-3
    end
end

println()
