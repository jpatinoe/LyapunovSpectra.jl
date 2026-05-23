# src/algorithm.jl

using LinearAlgebra

"""
    LyapunovProblem(v, J, d, k, β)

Defines a Lyapunov spectrum computation problem.

# Arguments
- `v`: vector field v(x) → ẋ  (in-place: v!(du, u))
- `J`: Jacobian J(x) → d×d matrix (in-place: J!(Jmat, u))
- `d`: dimension of the dynamical system
- `k`: number of Lyapunov exponents to compute  (1 ≤ k ≤ d)
- `β`: stability parameter. Must satisfy β > -λₖ (the k-th exponent).
       A safe default is β = 1.0; increase if orthonormality drifts.
"""
struct LyapunovProblem{F, G}
    v :: F
    J :: G
    d :: Int
    k :: Int
    β :: Float64
end

"""
    build_ode!(prob::LyapunovProblem)

Returns the ODE right-hand side function `f!(du, u, p, t)` compatible
with OrdinaryDiffEq.jl, encoding the augmented system (eq. 6 of the paper).

State vector layout (all concatenated into one flat vector):
  u[1:d]             → x      (trajectory)
  u[d+1:d+d*k]       → e₁…eₖ (frame, column-major)
  u[d+d*k+1:d+d*k+k] → Λ₁…Λₖ (accumulators)
"""
function build_ode!(prob::LyapunovProblem)

    d = prob.d
    k = prob.k
    β = prob.β
    v = prob.v
    J = prob.J

    # Pre-allocate ALL intermediate buffers here, outside the ODE function.
    # They are captured by the closure and reused on every ODE call.
    # This avoids heap allocation inside the hot loop.
    #
    # MATLAB note: in MATLAB you can't easily do this because functions
    # don't capture variables from an outer scope. In Julia, the inner
    # function `ode!` is a closure — it sees and reuses these buffers
    # automatically, like a persistent variable in MATLAB but cleaner.
    Jmat  = zeros(d, d)       # Jacobian matrix
    JE    = zeros(d, k)       # columns: J*e₁, J*e₂, …, J*eₖ

    function ode!(du, u, p, t)

        @views begin
            x  = u[1:d]
            E  = reshape(u[d+1 : d+d*k], d, k)
            Λ  = u[d+d*k+1 : d+d*k+k]
        end

        @views begin
            dx = du[1:d]
            dE = reshape(du[d+1 : d+d*k], d, k)
            dΛ = du[d+d*k+1 : d+d*k+k]
        end

        # Compute Jacobian at current point
        J(Jmat, x)

        # Trajectory equation: ẋ = v(x)
        v(dx, x)

        # Pre-compute ALL J*eₘ products at once — one matrix multiply
        # instead of k separate ones.
        # MATLAB note: this is Jmat * E, which gives a d×k matrix.
        # Each column m of JE is J*eₘ, reused in both the dΛ and dE equations.
        mul!(JE, Jmat, E)     # JE = Jmat * E,  in-place, no allocation

        for m in 1:k
            em  = E[:, m]
            Jem = JE[:, m]    # reuse pre-computed J*eₘ

            # Jₘₘ = (eₘ, J eₘ)
            Jmm = dot(em, Jem)

            # Accumulator equation: Λ̇ₘ = Jₘₘ
            dΛ[m] = Jmm

            # Frame equation: ėₘ = Jeₘ − Σ_{l≤m} eₗ Lₗₘ
            dE[:, m] .= Jem

            for l in 1:m
                el  = E[:, l]
                Jel = JE[:, l]   # reuse pre-computed J*eₗ — no allocation

                if l == m
                    Llm = Jmm + β * (dot(em, em) - 1.0)
                else
                    Jlm = dot(el, Jem)
                    Jml = dot(em, Jel)
                    Llm = Jlm + Jml + 2.0 * β * dot(el, em)
                end

                dE[:, m] .-= el .* Llm
            end
        end

        return nothing
    end

    return ode!
end

"""
    make_initial_state(prob, x0)

Build the flat initial state vector from an initial condition `x0`.

The frame E(0) is initialised to the identity matrix — the standard
basis vectors e₁, e₂, …, eₖ. This is valid because the theorem
guarantees convergence for almost any initial frame, and the identity
is a perfectly good choice in practice.

Λ(0) = 0 as required by the theorem.
"""
function make_initial_state(prob::LyapunovProblem, x0::Vector{Float64})
    d = prob.d
    k = prob.k

    # Identity frame: first k columns of the d×d identity matrix.
    # MATLAB note: this is eye(d, k) — the first k columns of the identity.
    E0 = Matrix{Float64}(I, d, k)

    return vcat(x0, vec(E0), zeros(k))
end