# =============================================================================
# WEAK FORMS 
# =============================================================================

# -----------------------------------------------------------------------------
# STRONG FORMS
# -----------------------------------------------------------------------------
# 
#   ∇·σ = 0  in Ω
#
# (Eq. 41):
#   (g_c/l)[d - l²Δd] = 2(1-d)H  in Ω


# -----------------------------------------------------------------------------
# WEAK FORM 1 — Mechanical equilibrium
# -----------------------------------------------------------------------------
# Multiply ∇·σ = 0 by test function δu, integrate over Ω:
#   ∫_Ω δu · (∇·σ) dV = 0
#
# Integrate by parts using product rule:
#   ∇·(σᵀδu) = (∇·σ)·δu + σ:∇δu
#   → (∇·σ)·δu = ∇·(σᵀδu) - σ:∇δu
#
# divergence theorem:
#   ∫_∂Ω δu·(σ·n) dA - ∫_Ω σ:∇δu dV = 0
#
# Split boundary integral into Dirichlet and Neumann parts:
#   ∫_∂Ω_u δu·(σ·n) dA + ∫_∂Ω_t δu·(σ·n) dA - ∫_Ω σ:∇δu dV = 0
#
# On ∂Ω_u: δu = 0 by definition of test function → first term vanishes
# On ∂Ω_t: σ·n = t̄ (Neumann BC) → second term becomes ∫_∂Ω_t δu·t̄ dA
#
# Rearrange into residual form, substitute σ = g(d)·σ₀, σ₀ = C:ε(u):
#
#   ∫_Ω g(d) σ₀(u) : ∇δu dV - ∫_∂Ω_t δu · t̄ dA = 0
#
# where g(d) = (1-d)² + k

# -----------------------------------------------------------------------------
# WEAK FORM 2 — Phase field evolution
# ---------------------------------------------------------------------------
# Rearrange strong form:
#   (g_c/l)d - g_c·l·Δd - 2(1-d)H = 0
#
# Multiply by scalar test function δd, integrate over Ω:
#   ∫_Ω (g_c/l) δd·d dV - ∫_Ω g_c·l·δd·Δd dV - ∫_Ω 2(1-d)H·δd dV = 0
#
# Only the middle term contains a spatial derivative of d (namely Δd = ∇·∇d)
# → product rule:
#   ∇·(δd·∇d) = ∇δd·∇d + δd·Δd
#   → δd·Δd = ∇·(δd·∇d) - ∇δd·∇d
#
# Apply divergence theorem to the middle term:
#   -∫_Ω g_c·l·δd·Δd dV = -∫_∂Ω g_c·l·δd·(∇d·n) dA + ∫_Ω g_c·l·∇δd·∇d dV
#
# Apply Neumann BC ∇d·n = 0 on ∂Ω → boundary term vanishes
#
# Final weak form:
#
#   ∫_Ω (g_c/l) δd·d dV + ∫_Ω g_c·l·∇δd·∇d dV - ∫_Ω 2(1-d)H·δd dV = 0
#

# -------------------------------------
# DISCRETISATION
# --------------------------------------
#
#  approximate d and δd using shape functions N over each element:
#
#   d(x)   ≈ Σ_j  Nⱼ(x) · dⱼ          (trial function)
#   δd(x)  ≈ Σ_i  Nᵢ(x) · δdᵢ         (test function)
#   ∇d(x)  ≈ Σ_j  ∇Nⱼ(x) · dⱼ
#   ∇δd(x) ≈ Σ_i  ∇Nᵢ(x) · δdᵢ

#   ke[i,j] = ∫_Ω (g_c/l) · Nᵢ · Nⱼ dV       ← from (g_c/2l)·d² term (×2 from differentiation)
#           + ∫_Ω g_c·l · ∇Nᵢ · ∇Nⱼ dV       ← from (g_c·l/2)·|∇d|² term (×2 from differentiation)
#           + ∫_Ω 2H · Nᵢ · Nⱼ dV             ← from -H·d² term (×2, sign flips to LHS)
#
#   fe[i]   = ∫_Ω 2H · Nᵢ dV                  ← from +2H·d linear term (moves to RHS)
# integrals are approximated by summing over quadrature points
# where dΩ = getdetJdV(cellvalues_d, qp) 

#NOTE:
# In Ferrite:
#   Nᵢ(x_qp)  = shape_value(cellvalues_d, qp, i)
#   ∇Nᵢ(x_qp) = shape_gradient(cellvalues_d, qp, i)
#   dΩ_qp     = getdetJdV(cellvalues_d, qp)
#   H(x_qp)   = H_cell[qp]


