# src/algorithm.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# THIS IS THE MATHEMATICAL CORE OF THE PACKAGE.
#
# This file implements the augmented dynamical system from equation (6) of:
#   Christiansen & Rugh (1997), Nonlinearity 10, 1063–1072.
#
# The key idea: instead of integrating the transition matrix M(t) directly
# (which fails numerically because its columns separate exponentially),
# we augment the original system with:
#   - an orthonormal k-frame E = {e₁, …, eₖ}  that tracks the k most
#     expanding directions while staying orthonormal at all times
#   - k accumulator scalars Λ₁, …, Λₖ  that collect the instantaneous
#     growth rates, so that Λₘ(T)/T → λₘ as T → ∞
#
# The full augmented state vector (one flat array) is:
#   u = [ x₁ … xd | e₁₁ … e₁d e₂₁ … ekd | Λ₁ … Λₖ ]
#         ───────   ─────────────────────   ─────────
#         d values       d×k values          k values
#
# ─────────────────────────────────────────────────────────────────────────────

using LinearAlgebra   # dot(), mul!(), I, zeros()


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURE: LyapunovProblem
#  Holds everything that defines a particular computation:
#  which system, how many exponents, and the stability parameter.
# ══════════════════════════════════════════════════════════════════════════════

"""
    LyapunovProblem(v, J, d, k, β)

Container that defines a Lyapunov spectrum computation problem.
Pass this struct to `lyapunov_exponents` to run the computation.

# Arguments
- `v`: vector field  v!(du, u)   — writes ẋ = v(x) into du  (in-place)
- `J`: Jacobian      J!(Jmat, u) — writes ∂v/∂x into Jmat   (in-place)
- `d`: dimension of the original dynamical system
- `k`: number of Lyapunov exponents to compute  (1 ≤ k ≤ d)
- `β`: stability parameter — must satisfy β > −λₖ  (see paper p.1065)
       Start with β = 1.0 and increase if orthonormality drifts during
       a trial run.
"""
struct LyapunovProblem{F, G}
    # F and G are "type parameters" — placeholders for whatever type
    # the functions v and J happen to be. This lets Julia compile
    # specialised, fast code for the exact functions you provide,
    # rather than using a slow generic dispatch.
    v :: F          # vector field function
    J :: G          # Jacobian function
    d :: Int        # system dimension
    k :: Int        # number of exponents requested
    β :: Float64    # stability parameter (paper: β > −λₖ, section 2)
end


# ══════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION: build_ode!
#
#  Given a LyapunovProblem, this function constructs the full augmented
#  system of equations (eq. 6 of the paper) automatically.
#  No need to write the augmented equations by hand.
#
#  It returns a function `ode!(du, u, p, t)` that the ODE solver calls
#  repeatedly to evaluate the right-hand side of the augmented system.
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_ode!(prob::LyapunovProblem)

Constructs and returns the right-hand side of the augmented ODE system
(equation 6 of Christiansen & Rugh 1997). This encodes:

  1. The original trajectory:   ẋ = v(x)
  2. The frame evolution:       ėₘ = Jeₘ − Σ_{l≤m} eₗ Lₗₘ
  3. The accumulator evolution: Λ̇ₘ = Jₘₘ

The returned function `ode!(du, u, p, t)` is compatible with
OrdinaryDiffEq.jl and can be passed directly to `ODEProblem`.
"""
function build_ode!(prob::LyapunovProblem)

    # ── Unpack the problem struct into local variables ────────────────────────
    
    d = prob.d
    k = prob.k
    β = prob.β
    v = prob.v
    J = prob.J

    # ── Pre-allocate  ─────────────────────────────────────────
    Jmat = zeros(d, d)   # the d×d Jacobian matrix ∂v/∂x
    JE   = zeros(d, k)   # d×k matrix: column m = J*eₘ

    # ── The ODE right-hand side ────────────────────────────────────────
    function ode!(du, u, p, t)

        # ── Unpack the state vector u into named views ────────────────
        # u is one long flat array. We interpret different sections as:
        #   x  — the trajectory point (d numbers)
        #   E  — the orthonormal frame (d×k matrix, stored column-major)
        #   Λ  — the Lyapunov accumulators (k numbers)
        #
        # `@views` makes these slices zero-copy "windows" into u — no new
        # memory is allocated. This is crucial for performance, since this
        # function is called at every time step. 
        @views begin
            x = u[1:d]                          # trajectory:   u[1..d]
            E = reshape(u[d+1 : d+d*k], d, k)  # frame:        u[d+1..d+dk], as d×k matrix
            Λ = u[d+d*k+1 : d+d*k+k]           # accumulators: u[d+dk+1..d+dk+k]
        end

        @views begin
            dx = du[1:d]                         # derivative of x
            dE = reshape(du[d+1 : d+d*k], d, k) # derivative of E (as d×k matrix)
            dΛ = du[d+d*k+1 : d+d*k+k]          # derivative of Λ
        end

        # ── Evaluate Jacobian at current trajectory point ─────────────
        # Fills Jmat = ∂v/∂x evaluated at x(t).
        # The Jacobian changes at every time step because x(t) moves.
        # We need it inside the ODE function, not pre-computed.
        J(Jmat, x)

        # ── Equation (6), line 1 — trajectory ─────────────────────────
        # ẋ = v(x)
        # This is just the original system. Nothing new here.
        v(dx, x)

        # ── Pre-compute all J*eₘ products at once ─────────────────────
        # JE[:,m] = Jmat * E[:,m] = J*eₘ  for all m simultaneously.
        # This is one d×d times d×k matrix multiply (fast, cache-friendly)
        # instead of k separate matrix-vector products.
        # We reuse these columns in both the frame equation and the
        # accumulator equation below — no redundant computation.
        #
        # mul!(C, A, B) computes C = A*B in-place (no allocation).
        mul!(JE, Jmat, E)

        # ── Equations (6), lines 2 & 3 — frame and accumulators ───────
        # Loop over each frame vector eₘ, m = 1, …, k.
        # The ordering m=1,2,…,k is essential: it mirrors the Gram-Schmidt
        # ordering that ensures λ₁ ≥ λ₂ ≥ … ≥ λₖ automatically.
        for m in 1:k

            em  = E[:, m]    # current frame vector eₘ  (a view, no copy)
            Jem = JE[:, m]   # J*eₘ, pre-computed above (a view, no copy)

            # Jₘₘ = (eₘ, J eₘ) — the instantaneous growth rate of eₘ.
            # This is the projection of J*eₘ onto eₘ itself.
            # Geometrically: how fast eₘ is being stretched along its own
            # direction by the flow. This is the continuous analogue of
            # log(‖ẽₘ‖) in the discrete Gram-Schmidt algorithm.
            Jmm = dot(em, Jem)

            # ── Equation (6), line 3: accumulator ─────────────────────────────
            # Λ̇ₘ = Jₘₘ
            # Integrating this gives Λₘ(T) = ∫₀ᵀ Jₘₘ dt, which is the
            # running total of growth in direction eₘ.
            # Dividing by T at the end gives the Lyapunov exponent λₘ.
            dΛ[m] = Jmm

            # ── Equation (6), line 2: frame — start with raw dynamics ──────────
            # ėₘ = J eₘ − Σ_{l≤m} eₗ Lₗₘ
            # We start by writing J*eₘ into dE[:,m], then subtract the
            # Gram-Schmidt and stabilisation terms in the inner loop below.
            dE[:, m] .= Jem

            # ── Inner loop: subtract projection terms (Gram-Schmidt + β) ───────
            # For each l from 1 to m, compute Lₗₘ and subtract eₗ * Lₗₘ.
            # This is the continuous Gram-Schmidt orthogonalisation.
            for l in 1:m

                el  = E[:, l]    # frame vector eₗ
                Jel = JE[:, l]   # J*eₗ, pre-computed above

                if l == m
                    # ── Diagonal term: l = m ──────────────────────────────────
                    # Lₘₘ = Jₘₘ + β·((eₘ,eₘ) − 1)
                    #
                    # Two parts:
                    #   Jₘₘ              — continuous normalisation.
                    #                      Subtracting eₘ·Jₘₘ from ėₘ keeps
                    #                      ‖eₘ‖ = 1 at all times. This is the
                    #                      continuous analogue of dividing by
                    #                      ‖ẽₘ‖ in discrete Gram-Schmidt.
                    #
                    #   β·((eₘ,eₘ) − 1) — stabilisation (this is Christiansen & Rugh's
                    #                      contribution).
                    #                      When ‖eₘ‖ = 1 exactly, this is zero.
                    #                      When numerical errors cause ‖eₘ‖ ≠ 1,
                    #                      this acts as a restoring spring force
                    #                      pulling eₘ back to unit length.
                    #                      β > 0 makes this strongly stable
                    #                      (paper Theorem, p.1065, Appendix A.3).
                    Llm = Jmm + β * (dot(em, em) - 1.0)

                else
                    # ── Off-diagonal term: l < m ──────────────────────────────
                    # Lₗₘ = Jₗₘ + Jₘₗ + 2β·(eₗ,eₘ)
                    #
                    # Three parts:
                    #   Jₗₘ = (eₗ, Jeₘ) — how much Jeₘ points along eₗ.
                    #                      Subtracting removes this component,
                    #                      maintaining (eₗ,eₘ) = 0.
                    #
                    #   Jₘₗ = (eₘ, Jeₗ) — symmetric correction. In discrete
                    #                      Gram-Schmidt only eₘ moves, so only
                    #                      Jₗₘ is needed. Here both eₗ AND eₘ
                    #                      move simultaneously, so we need to
                    #                      account for eₗ drifting toward eₘ
                    #                      as well. This ensures d/dt(eₗ,eₘ) = 0
                    #                      from both sides.
                    #
                    #   2β·(eₗ,eₘ)      — stabilisation (paper's contribution).
                    #                      When eₗ ⊥ eₘ exactly, this is zero.
                    #                      When numerical drift causes (eₗ,eₘ) ≠ 0,
                    #                      this pushes eₘ away from eₗ — a spring
                    #                      restoring orthogonality. It does NOT
                    #                      push eₘ toward any specific direction,
                    #                      only away from eₗ, so the theorem's
                    #                      "almost any frame" guarantee still holds.
                    Jlm = dot(el, Jem)   # (eₗ, J eₘ) — reuses pre-computed Jem
                    Jml = dot(em, Jel)   # (eₘ, J eₗ) — reuses pre-computed Jel
                    Llm = Jlm + Jml + 2.0 * β * dot(el, em)
                end

                # Subtract eₗ * Lₗₘ from ėₘ  (in-place, no allocation)
                # After the full loop, dE[:,m] = Jeₘ − Σ_{l≤m} eₗ Lₗₘ
                # which is exactly eq. (6), line 2 of the paper.
                dE[:, m] .-= el .* Llm
            end
        end

        return nothing   # function is called for its side effect on du
    end

    # Return the assembled ODE function.
    # The caller (solver.jl) passes this to ODEProblem for integration.
    return ode!
end


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: make_initial_state
#
#  Builds the full initial state vector u₀
#  from just the d-component initial condition x₀ for the trajectory.
#
#  The theorem (p.1064) requires:
#    - E(0) = any orthonormal frame  (almost any choice works)
#    - Λ(0) = 0
#  For simplicity and reproducibility, we choose the standard basis vectors
#  as the initial frame, and zero for the accumulators.
# ══════════════════════════════════════════════════════════════════════════════

"""
    make_initial_state(prob, x0)

Builds the flat initial state vector u₀ from the trajectory initial
condition x₀. Lays out u₀ as:

  u₀ = [ x₀ | vec(E₀) | 0…0 ]
         d      d×k       k

where E₀ = first k columns of the d×d identity matrix (the standard
basis), and Λ(0) = 0 as required by the theorem.

The identity frame is valid because the Oseledec theorem guarantees
convergence for almost any initial frame — the identity matrix is a
perfectly good choice in practice and avoids any randomness.
"""
function make_initial_state(prob::LyapunovProblem, x0::Vector{Float64})
    d = prob.d
    k = prob.k

    # Initial frame: first k columns of the d×d identity matrix.
    # These are the standard basis vectors e₁=(1,0,…,0), e₂=(0,1,…,0), …
    # They are orthonormal by construction, satisfying the theorem's requirement.
    E0 = Matrix{Float64}(I, d, k)

    # Concatenate into one flat state vector:
    #   x₀           → d numbers   (the trajectory initial condition)
    #   vec(E0)       → d*k numbers (the frame, stored column-by-column)
    #   zeros(k)      → k numbers   (accumulators start at 0, per the theorem)
    #
    return vcat(x0, vec(E0), zeros(k))
end
