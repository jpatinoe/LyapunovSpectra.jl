# LyapunovSpectra.jl

A Julia implementation of the continuous Gram–Schmidt orthonormalization
method for computing Lyapunov spectra, based on:

> F. Christiansen and H. H. Rugh,  
> *"Computing Lyapunov spectra with continuous Gram–Schmidt orthonormalization"*  
> Nonlinearity **10**, 1063–1072 (1997)

## Mathematical background

Given a dynamical system ẋ = v(x) in ℝᵈ, the Lyapunov spectrum
{λ₁ ≥ λ₂ ≥ … ≥ λₖ} describes the exponential growth rates of
infinitesimal perturbations along a trajectory.

The method augments the original system with an orthonormal k-frame
E = {e₁, …, eₖ} and a Lyapunov accumulator vector Λ:

```
ẋₘ  = v(x)
ėₘ  = J eₘ − Σ_{l≤m} eₗ Lₗₘ        m = 1, …, k
Λ̇ₘ  = Jₘₘ                           m = 1, …, k
```

where J = ∂v/∂x is the Jacobian, Jₘₘ = (eₘ, J eₘ), and Lₗₘ are
stabilized matrix elements:

```
Lₘₘ = Jₘₘ + β((eₘ,eₘ) − 1)
Lₗₘ = Jₗₘ + Jₘₗ + 2β(eₗ,eₘ)        l < m
```

The parameter β > 0 acts as a restoring force that continuously
corrects any drift from orthonormality. The Lyapunov exponents are
recovered as:

```
λₘ = lim_{t→∞} Λₘ(t) / t
```

The method is proven strongly stable when β > −λₖ (Theorem, p.1065).

## Repository structure

```
LyapunovSpectra.jl/
├── src/
│   ├── LyapunovSpectra.jl   # Module entry point
│   ├── algorithm.jl         # Core CGS augmented ODE (eq. 6)
│   ├── systems.jl           # Built-in example systems
│   └── solver.jl            # ODE wrapper and statistics
├── examples/
│   ├── lorenz.jl            # Reproduces Table 1 of the paper
│   └── hamiltonian.jl       # Reproduces Table 2 of the paper
├── test/
│   └── runtests.jl          # Automated test suite (8 tests)
└── Project.toml
```

## Installation

Clone the repository and instantiate the environment:

```bash
git clone <your-repo-url>
cd "Lyapunov Spectrum"
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

### Defining a system

You need to provide two functions:
- `v!(du, u)` — the vector field (in-place)
- `J!(Jmat, u)` — the Jacobian (in-place)

```julia
push!(LOAD_PATH, ".")
using LyapunovSpectra

# Example: Lorenz system
function my_v!(du, u)
    du[1] = 10.0*(u[2] - u[1])
    du[2] = 28.0*u[1] - u[2] - u[1]*u[3]
    du[3] = u[1]*u[2] - (8/3)*u[3]
end

function my_J!(J, u)
    J[1,1] = -10.0; J[1,2] =  10.0; J[1,3] =  0.0
    J[2,1] = 28.0 - u[3]; J[2,2] = -1.0; J[2,3] = -u[1]
    J[3,1] = u[2]; J[3,2] = u[1]; J[3,3] = -8/3
end

# Create the problem:
#   d = system dimension
#   k = number of exponents to compute (1 ≤ k ≤ d)
#   β = stability parameter (β > -λₖ; start with β=1 and increase if needed)
prob = LyapunovProblem(my_v!, my_J!, 3, 3, 20.0)
```

### Computing exponents

```julia
# Single initial condition
x0 = [1.0, 0.0, 0.0]
λ  = lyapunov_exponents(prob, x0, 1000.0)
# λ ≈ [0.906, 0.000, -14.572]

# Average over many initial conditions
x0_list = [randn(3) for _ in 1:100]
λ_mean, λ_std = lyapunov_exponents_mean(prob, x0_list, 1000.0)
```

### Using built-in systems

```julia
# Lorenz system (σ=10, r=28, b=8/3)
prob = lorenz_system()

# Quartic Hamiltonian (eq. 10 of the paper)
prob = hamiltonian_system()
```

### Choosing β

The stability parameter β must satisfy β > −λₖ where λₖ is the
smallest exponent you want to compute. In practice:

1. Start with β = 1.0
2. Run a short trial integration
3. If orthonormality drifts (‖E'E − I‖ grows), increase β
4. Do not set β too large — it increases the stiffness of the system

Values used in the paper: β = 20 (Lorenz), β = 0.5 (Hamiltonian).

## Running the examples

```bash
# Lorenz system — reproduces Table 1 of the paper
julia --project=. examples/lorenz.jl

# Quartic Hamiltonian — reproduces Table 2 of the paper
julia --project=. examples/hamiltonian.jl
```

## Running the tests

```bash
julia --project=. test/runtests.jl
```

Expected output: 8 tests, all passing.

## Validation

Two analytical constraints validate the algorithm independently of
the paper's numerical values:

**Lorenz:** The sum of all exponents equals the trace of the Jacobian
averaged over the attractor:
```
λ₁ + λ₂ + λ₃ = −σ − 1 − b = −13.6667
```

**Hamiltonian:** Symplectic structure forces exponents to come in pairs:
```
λₖ + λ_{d+1−k} = 0    for all k
```

Both constraints are satisfied to < 10⁻⁴ in our implementation.

## Dependencies

| Package | Role | 
|---|---|
| `OrdinaryDiffEq` | ODE integration |
| `LinearAlgebra` | `dot`, `norm`, `qr` |
| `Random` | Reproducible random frames |
| `Statistics` | `mean`, `std` over runs |
| `Printf` | Formatted output |

## Reference

Christiansen, F. and Rugh, H. H. (1997).
*Computing Lyapunov spectra with continuous Gram–Schmidt orthonormalization.*
Nonlinearity, 10(5), 1063–1072.