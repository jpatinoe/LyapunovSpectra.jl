# src/systems.jl
#
# Built-in dynamical systems for testing and examples.
# Each system provides:
#   v!(du, u)     — the vector field  (in-place)
#   J!(Jmat, u)   — the Jacobian      (in-place)
#   a suggested β, and a suggested initial condition

# ══════════════════════════════════════════════════════════════════════════════
#  LORENZ SYSTEM
#  ẋ = σ(y − x)
#  ẏ = rx − y − xz
#  ż = xy − bz
#  Standard parameters: σ=10, r=28, b=8/3
# ══════════════════════════════════════════════════════════════════════════════

"""
    lorenz_v!(du, u)

Vector field of the Lorenz system with σ=10, r=28, b=8/3.
Writes the result into `du` in-place (no return value needed).
"""
function lorenz_v!(du, u)
    σ = 10.0;  r = 28.0;  b = 8.0/3.0

    du[1] = σ * (u[2] - u[1])
    du[2] = r*u[1] - u[2] - u[1]*u[3]
    du[3] = u[1]*u[2] - b*u[3]

    return nothing
end

"""
    lorenz_J!(Jmat, u)

Analytical Jacobian of the Lorenz system.
Writes the d×d Jacobian matrix into `Jmat` in-place.

"""
function lorenz_J!(Jmat, u)
    σ = 10.0;  r = 28.0;  b = 8.0/3.0

    x, y, z = u[1], u[2], u[3]

    # The Jacobian ∂v/∂x is:
    #   [ -σ    σ    0  ]
    #   [ r-z  -1   -x  ]
    #   [  y    x   -b  ]

    Jmat[1,1] = -σ;    Jmat[1,2] =  σ;    Jmat[1,3] =  0.0
    Jmat[2,1] = r - z; Jmat[2,2] = -1.0;  Jmat[2,3] = -x
    Jmat[3,1] = y;     Jmat[3,2] =  x;    Jmat[3,3] = -b

    return nothing
end

"""
    lorenz_system()

Returns a `LyapunovProblem` for the Lorenz system ready to solve.
β=20 as chosen in the paper (section 3).
"""
function lorenz_system()
    return LyapunovProblem(lorenz_v!, lorenz_J!, 3, 3, 20.0)
end


# ══════════════════════════════════════════════════════════════════════════════
#  QUARTIC HAMILTONIAN  (eq. 10 in the paper)
#
#  H = (px² + py² + pz²)/2
#      + x²y² + y²z² + z²x²
#      + (x⁴ + y⁴ + z⁴)/32
#
#  State vector: u = [x, y, z, px, py, pz]   (d=6)
#  Hamilton's equations:
#    q̇ᵢ =  ∂H/∂pᵢ  →  ṗᵢ = pᵢ
#    ṗᵢ = −∂H/∂qᵢ  →  computed analytically below
# ══════════════════════════════════════════════════════════════════════════════

"""
    hamiltonian_v!(du, u)

Vector field of the quartic Hamiltonian system (eq. 10).
State: u = [x, y, z, px, py, pz]
"""
function hamiltonian_v!(du, u)
    x, y, z   = u[1], u[2], u[3]
    px, py, pz = u[4], u[5], u[6]

    # Position equations: q̇ = ∂H/∂p = p
    du[1] = px
    du[2] = py
    du[3] = pz

    # Momentum equations: ṗ = −∂H/∂q
    # ∂H/∂x = 2x*y² + 2z²*x + x³/8
    du[4] = -(2.0*x*y^2 + 2.0*z^2*x + x^3/8.0)
    # ∂H/∂y = 2x²*y + 2y*z² + y³/8
    du[5] = -(2.0*x^2*y + 2.0*y*z^2 + y^3/8.0)
    # ∂H/∂z = 2y²*z + 2z*x² + z³/8
    du[6] = -(2.0*y^2*z + 2.0*z*x^2 + z^3/8.0)

    return nothing
end

"""
    hamiltonian_J!(Jmat, u)

Analytical Jacobian of the quartic Hamiltonian system.

The Jacobian has the block structure:
   [ 0    I  ]
   [ -K   0  ]
where K = ∂²H/∂q² is the Hessian of H with respect to positions.
"""
function hamiltonian_J!(Jmat, u)
    x, y, z = u[1], u[2], u[3]

    fill!(Jmat, 0.0)

    # Top-right block: ∂(q̇)/∂p = I
    Jmat[1,4] = 1.0
    Jmat[2,5] = 1.0
    Jmat[3,6] = 1.0

    # Bottom-left block: ∂(ṗ)/∂q = −∂²H/∂q²  (negative Hessian)
    #
    # ∂²H/∂x² = 2y² + 2z² + 3x²/8
    Jmat[4,1] = -(2.0*y^2 + 2.0*z^2 + 3.0*x^2/8.0)
    # ∂²H/∂y² = 2x² + 2z² + 3y²/8
    Jmat[5,2] = -(2.0*x^2 + 2.0*z^2 + 3.0*y^2/8.0)
    # ∂²H/∂z² = 2y² + 2x² + 3z²/8
    Jmat[6,3] = -(2.0*y^2 + 2.0*x^2 + 3.0*z^2/8.0)

    # Off-diagonal: ∂²H/∂x∂y = 4xy
    Jmat[4,2] = -4.0*x*y;  Jmat[5,1] = -4.0*x*y
    # ∂²H/∂y∂z = 4yz
    Jmat[5,3] = -4.0*y*z;  Jmat[6,2] = -4.0*y*z
    # ∂²H/∂x∂z = 4xz
    Jmat[4,3] = -4.0*x*z;  Jmat[6,1] = -4.0*x*z

    return nothing
end

"""
    hamiltonian_system()

Returns a `LyapunovProblem` for the quartic Hamiltonian.
β=0.5 as chosen in the paper (section 3).
"""
function hamiltonian_system()
    return LyapunovProblem(hamiltonian_v!, hamiltonian_J!, 6, 6, 0.5)
end