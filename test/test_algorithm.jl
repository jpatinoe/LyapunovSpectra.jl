# test/test_algorithm.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# GENERIC ALGORITHM TESTS
#
# Provides two callable functions.
#
# test_dissipative(prob, λ, state)  — properties that hold for ANY system
# test_hamiltonian(prob, λ, state)  — calls test_dissipative, then adds
#                                     Hamiltonian-specific checks
#
# Both functions take:
#   prob  — a LyapunovProblem
#   λ     — the Lyapunov exponents from lyapunov_exponents(...; return_state=true)
#   state — the LyapunovState returned alongside λ
#           (needed for D3 frame orthonormality — avoids re-integrating)
# ─────────────────────────────────────────────────────────────────────────────

using Test
using Printf
using LinearAlgebra


"""
    test_dissipative(prob, λ, state; atol_order, atol_zero, atol_ortho, atol_partial)

Run generic algorithm tests for any dynamical system.

# Tests performed
- D1: exponents in descending order  λ₁ ≥ λ₂ ≥ … ≥ λₖ
- D2: one exponent near zero         (flow direction of any autonomous system)
- D3: frame orthonormality           ‖E'E − I‖ < atol_ortho  (uses saved state)
- D4: k=1 consistent with full       λ₁(k=1) ≈ λ₁(k=d)      (one extra integration)
"""
function test_dissipative(prob, λ, state;
                          atol_order   = 1e-10,
                          atol_zero    = 0.05,
                          atol_ortho   = 1e-5,
                          atol_partial = 0.05)

    k = prob.k
    d = prob.d
    T = state.T_total

    # ── D1: Exponents in descending order ─────────────────────────────────────
    # Structural property of the Gram-Schmidt ordering — needs only λ.
    @testset "D1: Exponents in descending order" begin
        println("  D1: Descending order")
        for m in 1:k-1
            @printf "      λ%d = %+.4f ≥ λ%d = %+.4f\n" m λ[m] (m+1) λ[m+1]
            @test λ[m] ≥ λ[m+1] - atol_order
        end
    end

    # ── D2: One exponent near zero ────────────────────────────────────────────
    # For any autonomous system, the flow direction is neutral — needs only λ.
    @testset "D2: At least one exponent near zero (flow direction)" begin
        println("  D2: At least one exponent near zero")
        min_abs = minimum(abs.(λ))
        @printf "      min|λₘ| = %.6f  (threshold: %.2f)\n" min_abs atol_zero
        @test min_abs < atol_zero
    end

    # ── D3: Frame orthonormality ──────────────────────────────────────────────
    # Uses the saved state — no re-integration needed.
    # Extract E from the final augmented state vector u(T).
    @testset "D3: Frame orthonormality ‖E'E − I‖ < $atol_ortho" begin
        println("  D3: Frame orthonormality")
        u_final = state.u
        E_final = reshape(u_final[d+1 : d+d*k], d, k)
        err     = norm(E_final' * E_final - I)
        @printf "      ‖E'E − I‖ = %.2e  (threshold: %.0e)\n" err atol_ortho
        @test err < atol_ortho
    end

    # ── D4: Partial spectrum (k=1) consistent with full spectrum ──────────────
    # Needs one extra integration with k=1.
    # We use x(T) from the saved state as the initial condition so both
    # runs start from the same trajectory point, reducing variability.
    @testset "D4: Partial spectrum (k=1) consistent with full spectrum" begin
        println("  D4: k=1 consistent with full spectrum")
        x0_restart   = state.u[1:d]                  # trajectory at end of run
        prob_partial = LyapunovProblem(prob.v, prob.J, d, 1, prob.β)
        λ_partial    = lyapunov_exponents(prob_partial, x0_restart, T)
        @printf "      λ₁ (k=%d) = %+.4f\n" k λ[1]
        @printf "      λ₁ (k=1)  = %+.4f  (tolerance: ±%.2f)\n" λ_partial[1] atol_partial
        @test λ[1] ≈ λ_partial[1] atol=atol_partial
    end
end


"""
    test_hamiltonian(prob, λ, state; atol_symplectic, atol_sum, kwargs...)

Run generic algorithm tests for a Hamiltonian system.
Calls `test_dissipative` first, then adds Hamiltonian-specific checks.

# Additional tests
- H1: symplectic pairing  λₖ + λ_{d+1−k} ≈ 0  for all k  (needs only λ)
- H2: sum of exponents ≈ 0  (tr(J) = 0 for any Hamiltonian) (needs only λ)
"""
function test_hamiltonian(prob, λ, state;
                          atol_symplectic = 1e-3,
                          atol_sum        = 1e-3,
                          kwargs...)

    # Run all dissipative tests first
    test_dissipative(prob, λ, state; kwargs...)

    d = prob.d

    # ── H1: Symplectic pairing ────────────────────────────────────────────────
    # Fundamental property of Hamiltonian dynamics — needs only λ.
    @testset "H1: Symplectic pairing λₖ + λ_{d+1-k} ≈ 0" begin
        println("  H1: Symplectic pairing")
        for i in 1:d÷2
            pair_sum = λ[i] + λ[d+1-i]
            @printf "      λ%d + λ%d = %+.6f  (threshold: %.0e)\n" i (d+1-i) pair_sum atol_symplectic
            @test abs(pair_sum) < atol_symplectic
        end
    end

    # ── H2: Sum of exponents near zero ────────────────────────────────────────
    # tr(J) = 0 for any Hamiltonian system — needs only λ.
    @testset "H2: Sum of exponents ≈ 0 (volume-preserving flow)" begin
        println("  H2: Sum of exponents ≈ 0")
        s = sum(λ)
        @printf "      Σλₘ = %+.6f  (threshold: %.0e)\n" s atol_sum
        @test abs(s) < atol_sum
    end
end
