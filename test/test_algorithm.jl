# test/test_algorithm.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# GENERIC ALGORITHM TESTS
#
# This file provides two callable functions — not a standalone script.
# Call them from test_examples.jl (or your own test file) after computing
# the Lyapunov exponents for any system.
#
# test_dissipative(prob, λ, x0, T)  — properties that hold for ANY system
# test_hamiltonian(prob, λ, x0, T)  — calls test_dissipative, then adds
#                                     Hamiltonian-specific checks
#
# Both functions take:
#   prob  — a LyapunovProblem (needed for frame orthonormality and k=1 test)
#   λ     — the Lyapunov exponents already computed by lyapunov_exponents()
#   x0    — the initial condition used to compute λ
#   T     — the integration time used to compute λ
#
# Tolerances can be overridden via keyword arguments — useful when applying
# to a new system where the defaults may be too tight or too loose.
# ─────────────────────────────────────────────────────────────────────────────

using Test
using Printf
using LinearAlgebra
using OrdinaryDiffEq


"""
    test_dissipative(prob, λ, x0, T; atol_order, atol_zero, atol_ortho, atol_partial)

Run generic algorithm tests for any dynamical system.
Call this from your own test file after computing λ = lyapunov_exponents(prob, x0, T).

# Tests performed
- D1: exponents in descending order  λ₁ ≥ λ₂ ≥ … ≥ λₖ
- D2: one exponent near zero         (flow direction of any autonomous system)
- D3: frame orthonormality           ‖E'E − I‖ < atol_ortho
- D4: k=1 consistent with full       λ₁(k=1) ≈ λ₁(k=d)

# Keyword arguments
- `atol_order`   : tolerance for descending order check (default: 1e-10)
- `atol_zero`    : tolerance for the near-zero exponent (default: 0.05)
- `atol_ortho`   : tolerance for frame orthonormality   (default: 1e-6)
- `atol_partial` : tolerance for k=1 vs full spectrum   (default: 0.05)
"""
function test_dissipative(prob, λ, x0, T;
                          atol_order   = 1e-10,
                          atol_zero    = 0.05,
                          atol_ortho   = 1e-6,
                          atol_partial = 0.05)

    k = prob.k
    d = prob.d

    # ── D1: Exponents in descending order ─────────────────────────────────────
    # The Gram-Schmidt ordering guarantees λ₁ ≥ λ₂ ≥ … ≥ λₖ for any system.
    # This is a structural property of the algorithm — if this fails,
    # something is wrong with the frame evolution in algorithm.jl.
    @testset "D1: Exponents in descending order" begin
        println("  D1: Descending order")
        for m in 1:k-1
            @printf "      λ%d = %+.4f ≥ λ%d = %+.4f\n" m λ[m] (m+1) λ[m+1]
            @test λ[m] ≥ λ[m+1] - atol_order
        end
    end

    # ── D2: One exponent near zero ────────────────────────────────────────────
    # For any autonomous system, the direction along the flow neither grows
    # nor shrinks — so one Lyapunov exponent must be (approximately) zero.
    # We check that the minimum absolute value across all exponents is small.
    # Note: for Hamiltonian systems there are two zeros (flow + energy), but
    # checking for at least one is sufficient here.
    @testset "D2: At least one exponent near zero (flow direction)" begin
        println("  D2: At least one exponent near zero")
        min_abs = minimum(abs.(λ))
        @printf "      min|λₘ| = %.6f  (threshold: %.2f)\n" min_abs atol_zero
        @test min_abs < atol_zero
    end

    # ── D3: Frame orthonormality ──────────────────────────────────────────────
    # The β stabilisation term in eq. (6) keeps the frame orthonormal.
    # We rebuild the solution with save_everystep=false and check ‖E'E − I‖
    # at the final time. This tests the stabilisation mechanism directly.
    @testset "D3: Frame orthonormality ‖E'E − I‖ < $atol_ortho" begin
        println("  D3: Frame orthonormality")
        u0       = make_initial_state(prob, x0)
        ode!     = build_ode!(prob)
        ode_prob = ODEProblem(ode!, u0, (0.0, T), nothing)
        sol      = solve(ode_prob, Tsit5();
                         reltol         = 1e-8,
                         abstol         = 1e-8,
                         maxiters       = 1_000_000,
                         save_everystep = false)

        E_final = reshape(sol.u[end][d+1 : d+d*k], d, k)
        err     = norm(E_final' * E_final - I)

        @printf "      ‖E'E − I‖ = %.2e  (threshold: %.0e)\n" err atol_ortho
        @test err < atol_ortho
    end

    # ── D4: Partial spectrum (k=1) consistent with full spectrum ──────────────
    # Computing only k=1 should give the same λ₁ as the full k=d computation.
    # This tests that reducing k does not corrupt the largest exponent.
    # We use a loose tolerance because two independent integrations accumulate
    # different floating-point rounding errors.
    @testset "D4: Partial spectrum (k=1) consistent with full spectrum" begin
        println("  D4: k=1 consistent with full spectrum")
        prob_partial = LyapunovProblem(prob.v, prob.J, prob.d, 1, prob.β)
        λ_partial    = lyapunov_exponents(prob_partial, x0, T)
        @printf "      λ₁ (k=%d) = %+.4f\n" k λ[1]
        @printf "      λ₁ (k=1)  = %+.4f  (tolerance: ±%.2f)\n" λ_partial[1] atol_partial
        @test λ[1] ≈ λ_partial[1] atol=atol_partial
    end
end


"""
    test_hamiltonian(prob, λ, x0, T; kwargs...)

Run generic algorithm tests for a Hamiltonian system.
Calls `test_dissipative` first, then adds Hamiltonian-specific checks.

# Additional tests performed
- H1: symplectic pairing  λₖ + λ_{d+1−k} ≈ 0  for all k
- H2: sum of exponents ≈ 0  (tr(J) = 0 for any Hamiltonian system)

All keyword arguments are forwarded to `test_dissipative`.
Additional keyword:
- `atol_symplectic` : tolerance for symplectic pairs (default: 1e-3)
- `atol_sum`        : tolerance for sum ≈ 0         (default: 1e-3)
"""
function test_hamiltonian(prob, λ, x0, T;
                          atol_symplectic = 1e-3,
                          atol_sum        = 1e-3,
                          kwargs...)

    # Run all dissipative tests first
    test_dissipative(prob, λ, x0, T; kwargs...)

    d = prob.d
    k = prob.k

    # ── H1: Symplectic pairing ────────────────────────────────────────────────
    # For any Hamiltonian system, symplectic structure forces exponents to
    # come in conjugate pairs: λₖ + λ_{d+1−k} = 0 for all k.
    # This is a fundamental property of Hamiltonian dynamics — if this fails,
    # the frame dynamics are not preserving the symplectic structure.
    # Requires k = d (full spectrum).
    @testset "H1: Symplectic pairing λₖ + λ_{d+1-k} ≈ 0" begin
        println("  H1: Symplectic pairing")
        for i in 1:d÷2
            pair_sum = λ[i] + λ[d+1-i]
            @printf "      λ%d + λ%d = %+.6f  (threshold: %.0e)\n" i (d+1-i) pair_sum atol_symplectic
            @test abs(pair_sum) < atol_symplectic
        end
    end

    # ── H2: Sum of exponents near zero ────────────────────────────────────────
    # For any Hamiltonian system tr(J) = 0 everywhere (the flow is
    # volume-preserving), so the sum of all Lyapunov exponents must be zero.
    # This follows directly from H1 but is a useful independent check.
    @testset "H2: Sum of exponents ≈ 0 (volume-preserving flow)" begin
        println("  H2: Sum of exponents ≈ 0")
        s = sum(λ)
        @printf "      Σλₘ = %+.6f  (threshold: %.0e)\n" s atol_sum
        @test abs(s) < atol_sum
    end
end
