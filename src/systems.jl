# src/systems.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE SYSTEMS.
#
# This file contains the two dynamical systems used in the paper.
#
# Each system requires exactly two functions:
#   v!(du, u)     — the vector field:  writes ẋ = v(x) into du
#   J!(Jmat, u)   — the Jacobian:      writes ∂v/∂x into Jmat
#
# Both are "in-place": they write into a pre-allocated output buffer
# rather than returning a new array. This avoids memory allocation
# inside the ODE solver's hot loop.
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM 1: LORENZ  (paper section 3)
#
#  The classic chaotic system in R³:
#    ẋ = σ(y − x)
#    ẏ = rx − y − xz
#    ż = xy − bz
#
#  Standard parameters: σ=10, r=28, b=8/3
#  Dimension: d=3, so the augmented system has 3 + 9 + 3 = 15 equations.
# ══════════════════════════════════════════════════════════════════════════════

"""
    lorenz_v!(du, u)

Vector field of the Lorenz system with σ=10, r=28, b=8/3.
Writes ẋ=v(x) into `du` in-place.

State vector: u = [x, y, z]
"""
function lorenz_v!(du, u)
    σ = 10.0;  r = 28.0;  b = 8.0/3.0

    # The three Lorenz equations 
    du[1] = σ * (u[2] - u[1])
    du[2] = r*u[1] - u[2] - u[1]*u[3]
    du[3] = u[1]*u[2] - b*u[3]

    return nothing
end

"""
    lorenz_J!(Jmat, u)

Analytical Jacobian ∂v/∂x of the Lorenz system.
Writes the 3×3 Jacobian matrix into `Jmat` in-place.

The Jacobian is:
  ∂v/∂x = [ -σ     σ    0  ]
           [ r-z   -1   -x  ]
           [  y     x   -b  ]

This is used by build_ode! to compute Jₘₘ = (eₘ, J eₘ) and the
stabilised projection terms Lₗₘ at each time step.
"""
function lorenz_J!(Jmat, u)
    σ = 10.0;  r = 28.0;  b = 8.0/3.0
    x, y, z = u[1], u[2], u[3]

    # Fill the Jacobian matrix entry by entry.
    # Here we write in-place to avoid allocating a new 3×3 matrix every call.
    Jmat[1,1] = -σ;    Jmat[1,2] =  σ;    Jmat[1,3] =  0.0
    Jmat[2,1] = r - z; Jmat[2,2] = -1.0;  Jmat[2,3] = -x
    Jmat[3,1] = y;     Jmat[3,2] =  x;    Jmat[3,3] = -b

    return nothing
end

"""
    lorenz_system()

Returns a ready-to-use `LyapunovProblem` for the Lorenz system.

Parameters match the paper:
  d=3  (3-dimensional system)
  k=3  (compute the full spectrum of 3 exponents)
  β=20 (stability parameter, chosen in paper section 3)
"""
function lorenz_system()
    # β=20 was chosen in the paper
    # it is well above −λ₃ ≈ 14.6, ensuring strong stability.
    return LyapunovProblem(lorenz_v!, lorenz_J!, 3, 3, 20.0)
end

# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM 2: QUARTIC HAMILTONIAN  (paper section 3)
#
#  A 3-degree-of-freedom Hamiltonian system:
#    H = (px² + py² + pz²)/2
#        + x²y² + y²z² + z²x²
#        + (x⁴ + y⁴ + z⁴)/32
#
#  State vector: u = [x, y, z, px, py, pz]   →   d=6
#  Hamilton's equations give 6 ODEs:
#    q̇ᵢ = +∂H/∂pᵢ = pᵢ              (positions evolve as momenta)
#    ṗᵢ = −∂H/∂qᵢ                   (momenta evolve as negative potential gradient)
#
#  The quartic term (x⁴+y⁴+z⁴)/32 is added with a small prefactor to
#  make the phase space compact (bounded trajectories) without over-confining
#  the dynamics — see paper footnote on eq. (10).
#
#  Dimension: d=6, so the augmented system has 6 + 36 + 6 = 48 equations.
#
#  Validation: symplectic structure forces λₖ + λ_{7−k} = 0 for all k.
#  This means λ₁=−λ₆, λ₂=−λ₅, λ₃=−λ₄ — exponents come in ±pairs.
# ══════════════════════════════════════════════════════════════════════════════

"""
    hamiltonian_v!(du, u)

Vector field of the quartic Hamiltonian system (paper eq. 10).
Writes ẋ=v(x) into `du` in-place.

State vector: u = [x, y, z, px, py, pz]
"""
function hamiltonian_v!(du, u)
    x, y, z    = u[1], u[2], u[3]
    px, py, pz = u[4], u[5], u[6]

    # Position equations: q̇ᵢ = ∂H/∂pᵢ = pᵢ
    # Positions evolve simply as their conjugate momenta.
    du[1] = px
    du[2] = py
    du[3] = pz

    # Momentum equations: ṗᵢ = −∂H/∂qᵢ
    # Computed by differentiating H with respect to each position coordinate.
    #
    # ∂H/∂x = 2xy² + 2xz² + x³/8
    du[4] = -(2.0*x*y^2 + 2.0*z^2*x + x^3/8.0)
    # ∂H/∂y = 2x²y + 2yz² + y³/8
    du[5] = -(2.0*x^2*y + 2.0*y*z^2 + y^3/8.0)
    # ∂H/∂z = 2y²z + 2x²z + z³/8
    du[6] = -(2.0*y^2*z + 2.0*z*x^2 + z^3/8.0)

    return nothing
end

"""
    hamiltonian_J!(Jmat, u)

Analytical Jacobian ∂v/∂x of the quartic Hamiltonian system.
Writes the 6×6 Jacobian matrix into `Jmat` in-place.

The Jacobian has the canonical Hamiltonian block structure:
  ∂v/∂(q,p) = [  0    I  ]
               [ -K    0  ]
where:
  I  = 3×3 identity  (top-right block: positions respond to momenta)
  -K = −∂²H/∂q²      (bottom-left block: momenta respond to positions
                       via the negative Hessian of H)
  0  = 3×3 zeros     (top-left and bottom-right: no self-coupling)

This block structure is the source of the symplectic pairing of
Lyapunov exponents: λₖ + λ_{7−k} = 0.
"""
function hamiltonian_J!(Jmat, u)
    x, y, z = u[1], u[2], u[3]

    # Start from zero — only the non-zero blocks will be filled.
    fill!(Jmat, 0.0)

    # ── Top-right block: ∂(q̇)/∂p = I ─────────────────────────────────────────
    # Positions respond to momenta with unit gain (q̇ = p).
    Jmat[1,4] = 1.0
    Jmat[2,5] = 1.0
    Jmat[3,6] = 1.0

    # ── Bottom-left block: ∂(ṗ)/∂q = −∂²H/∂q² ────────────────────────────────
    # Diagonal entries: second derivatives of H with respect to each position.
    # ∂²H/∂x² = 2y² + 2z² + 3x²/8
    Jmat[4,1] = -(2.0*y^2 + 2.0*z^2 + 3.0*x^2/8.0)
    # ∂²H/∂y² = 2x² + 2z² + 3y²/8
    Jmat[5,2] = -(2.0*x^2 + 2.0*z^2 + 3.0*y^2/8.0)
    # ∂²H/∂z² = 2y² + 2x² + 3z²/8
    Jmat[6,3] = -(2.0*y^2 + 2.0*x^2 + 3.0*z^2/8.0)

    # Off-diagonal entries: mixed second derivatives (cross terms in H).
    # The Hessian is symmetric, so each cross derivative appears twice.
    # ∂²H/∂x∂y = 4xy  →  affects both (4,2) and (5,1)
    Jmat[4,2] = -4.0*x*y;  Jmat[5,1] = -4.0*x*y
    # ∂²H/∂y∂z = 4yz  →  affects both (5,3) and (6,2)
    Jmat[5,3] = -4.0*y*z;  Jmat[6,2] = -4.0*y*z
    # ∂²H/∂x∂z = 4xz  →  affects both (4,3) and (6,1)
    Jmat[4,3] = -4.0*x*z;  Jmat[6,1] = -4.0*x*z

    return nothing
end

"""
    hamiltonian_system()

Returns a ready-to-use `LyapunovProblem` for the quartic Hamiltonian system.

Parameters match the paper:
  d=6  (6-dimensional phase space: 3 positions + 3 momenta)
  k=6  (compute the full spectrum of 6 exponents)
  β=0.5 (stability parameter, chosen in paper section 3)
"""
function hamiltonian_system()
    # β=0.5 was chosen in the paper.
    return LyapunovProblem(hamiltonian_v!, hamiltonian_J!, 6, 6, 0.5)
end
