# test/runtests.jl
#
# Automated test suite for LyapunovSpectra.jl
# Run with:
#   julia --project=. test/runtests.jl
#
# MATLAB note: Julia has a built-in testing framework called `Test`.
# It works like assert() in MATLAB but gives much more informative
# output. The macros are:
#   @test expr        — passes if expr is true (like assert in MATLAB)
#   @testset "name"   — groups related tests together
#   @test a ≈ b       — passes if a and b are approximately equal
#                       (uses atol and rtol, like assertAlmostEqual)

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using LyapunovSpectra
using Test
using Printf

println("="^60)
println("  LyapunovSpectra.jl — Test Suite")
println("="^60)

# ── Test 1: Lorenz sum constraint ─────────────────────────────────────────────
# The sum of all Lyapunov exponents must equal -σ - 1 - b = -13.6667
# This is an EXACT analytical result — any correct implementation must hit it.
# MATLAB note: @testset groups tests like a test class in MATLAB's matlab.unittest
@testset "Lorenz: sum of exponents" begin
    println("\nTest 1: Lorenz sum constraint (λ₁+λ₂+λ₃ = -13.6667)")

    prob = lorenz_system()
    x0   = [1.0, 0.0, 0.0]
    λ    = lyapunov_exponents(prob, x0, 2000.0)

    expected_sum = -(10.0 + 1.0 + 8.0/3.0)   # -σ - 1 - b
    actual_sum   = sum(λ)

    @printf "  λ₁=%+.4f  λ₂=%+.4f  λ₃=%+.4f\n" λ[1] λ[2] λ[3]
    @printf "  Sum = %.6f  (expected %.6f)\n" actual_sum expected_sum

    # MATLAB note: ≈ is the approximate equality operator.
    # atol=1e-4 means we accept |actual - expected| < 0.0001
    @test actual_sum ≈ expected_sum atol=1e-4
end

# ── Test 2: Lorenz largest exponent ───────────────────────────────────────────
# λ₁ should be positive (the system is chaotic)
# and converge toward 0.9057 for long T
@testset "Lorenz: largest exponent is positive" begin
    println("\nTest 2: Lorenz largest exponent is positive")

    prob = lorenz_system()
    x0   = [1.0, 0.0, 0.0]
    λ    = lyapunov_exponents(prob, x0, 2000.0)

    @printf "  λ₁ = %+.4f  (should be > 0)\n" λ[1]
    @test λ[1] > 0.0
end

# ── Test 3: Lorenz second exponent near zero ───────────────────────────────────
# λ₂ corresponds to the flow direction — must be ~0
@testset "Lorenz: second exponent near zero" begin
    println("\nTest 3: Lorenz second exponent near zero")

    prob = lorenz_system()
    x0   = [1.0, 0.0, 0.0]
    λ    = lyapunov_exponents(prob, x0, 2000.0)

    @printf "  λ₂ = %+.6f  (should be ~0)\n" λ[2]
    @test abs(λ[2]) < 0.01
end

# ── Test 4: Lorenz exponents in descending order ───────────────────────────────
# The solver must return exponents sorted λ₁ ≥ λ₂ ≥ λ₃
@testset "Lorenz: exponents in descending order" begin
    println("\nTest 4: Lorenz exponents in descending order")

    prob = lorenz_system()
    x0   = [1.0, 0.0, 0.0]
    λ    = lyapunov_exponents(prob, x0, 2000.0)

    @printf "  λ₁=%+.4f ≥ λ₂=%+.4f ≥ λ₃=%+.4f\n" λ[1] λ[2] λ[3]
    @test λ[1] ≥ λ[2]
    @test λ[2] ≥ λ[3]
end

# ── Test 5: Hamiltonian symplectic pairs ───────────────────────────────────────
# For a Hamiltonian system λₖ + λ_{d+1-k} = 0 exactly.
# This is the strongest validation for the Hamiltonian case.
@testset "Hamiltonian: symplectic pairs sum to zero" begin
    println("\nTest 5: Hamiltonian symplectic pairs (λₖ + λ_{7-k} ≈ 0)")

    prob = hamiltonian_system()
    x0   = [0.5, 0.3, 0.7, 0.3, 0.2, 0.4]
    λ    = lyapunov_exponents(prob, x0, 5000.0)

    for k in 1:3
        pair_sum = λ[k] + λ[7-k]
        @printf "  λ%d + λ%d = %+.6f  (should be ~0)\n" k (7-k) pair_sum
        @test abs(pair_sum) < 1e-3
    end
end

# ── Test 6: Hamiltonian largest exponent is positive ──────────────────────────
@testset "Hamiltonian: largest exponent is positive" begin
    println("\nTest 6: Hamiltonian largest exponent is positive")

    prob = hamiltonian_system()
    x0   = [0.5, 0.3, 0.7, 0.3, 0.2, 0.4]
    λ    = lyapunov_exponents(prob, x0, 2000.0)

    @printf "  λ₁ = %+.4f  (should be > 0)\n" λ[1]
    @test λ[1] > 0.0
end

# ── Test 7: Partial spectrum ───────────────────────────────────────────────────
# Compute only k=1 exponent — should match λ₁ from full spectrum
@testset "Partial spectrum (k=1) matches full spectrum" begin
    println("\nTest 7: Partial spectrum k=1 matches full spectrum")

    x0   = [1.0, 0.0, 0.0]

    # Full spectrum (k=3)
    prob_full    = lorenz_system()
    λ_full       = lyapunov_exponents(prob_full, x0, 2000.0)

    # Partial spectrum (k=1) — same β, same system
    prob_partial = LyapunovProblem(prob_full.v, prob_full.J, 3, 1, prob_full.β)
    λ_partial    = lyapunov_exponents(prob_partial, x0, 2000.0)

    @printf "  λ₁ full    = %+.4f\n" λ_full[1]
    @printf "  λ₁ partial = %+.4f\n" λ_partial[1]
    @test λ_full[1] ≈ λ_partial[1] atol=0.05
end

# ── Test 8: Frame orthonormality ───────────────────────────────────────────────
# The β stabilization should keep the frame orthonormal throughout.
# We check this by inspecting the final state vector directly.
@testset "Frame stays orthonormal" begin
    println("\nTest 8: Frame orthonormality at end of integration")

    using LinearAlgebra

    prob = lorenz_system()
    x0   = [1.0, 0.0, 0.0]
    d    = prob.d
    k    = prob.k

    # Build and solve the augmented system manually to access final frame
    u0      = make_initial_state(prob, x0)
    ode!    = build_ode!(prob)

    using OrdinaryDiffEq
    ode_prob = ODEProblem(ode!, u0, (0.0, 2000.0), nothing)
    sol      = solve(ode_prob, Tsit5();
                     reltol=1e-8, abstol=1e-8,
                     maxiters=1_000_000,
                     save_everystep=false)

    # Extract final frame E (d×k matrix)
    u_final = sol.u[end]
    E_final = reshape(u_final[d+1 : d+d*k], d, k)

    # E'E should be the identity matrix — like orth() check in MATLAB
    # MATLAB note: E'*E is just A'*A in MATLAB
    EtE  = E_final' * E_final
    err  = norm(EtE - I)   # I is the identity matrix from LinearAlgebra

    @printf "  ||E'E - I|| = %.2e  (should be < 1e-6)\n" err
    @test err < 1e-6
end

println("\n" * "="^60)
println("  All tests complete.")
println("="^60)