# Equation 17 — elastoplastic tangent tensor C_ep
function compute_Cep(σ::SymmetricTensor{2, 2}, mat::LinearElasticMaterial, h::Float64)
    n     = flow_direction(σ)
    C     = mat.C
    Cn    = C ⊡ n                        # = n ⊡ C since C is major-symmetric
    denom = (n ⊡ Cn) + h                 # scalar: n:C:n + h
    return C - otimes(Cn, Cn) / denom    # SymmetricTensor{4,2}
end