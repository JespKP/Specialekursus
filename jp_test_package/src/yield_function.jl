# Equation 8 — effective stress (the scalar σ_e)
function effective_stress(σ::SymmetricTensor{2, 2})
    σ11, σ22, σ12 = σ[1,1], σ[2,2], σ[1,2]
    sqrt(σ11^2 + σ22^2 - σ11*σ22 + 3/2 * σ12^2 + 3/2 * σ12^2)
end

# Equation 8 — yield function F = σ_e - σ_y
function yield_function(σ::SymmetricTensor{2, 2}, σ_y::Float64)
    effective_stress(σ) - σ_y
end

# Equation 9 — flow direction ∂F/∂σ (returns a SymmetricTensor)
function flow_direction(σ::SymmetricTensor{2, 2})
    σ11, σ22, σ12 = σ[1,1], σ[2,2], σ[1,2]
    σ_e = effective_stress(σ)
    if σ_e == 0.0
        return zero(SymmetricTensor{2, 2})
    end    


    SymmetricTensor{2, 2}((
        (2*σ11 - σ22) / (2*σ_e),
        (2*σ22 - σ11) / (2*σ_e),
        (6*σ12)       / (2*σ_e)   # the 2ε12 factor
    ))
end