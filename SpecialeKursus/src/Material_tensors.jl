
function plane_strain_tensors(Emod, nu)
    Gmod = Emod / (2 * (1 + nu))
    Kmod = Emod * nu / ((1 + nu) * (1 - 2 * nu))

    C_dil = gradient(eps -> Kmod * tr(eps) * one(eps), zero(SymmetricTensor{2, 2}))
    C_dev = gradient(eps -> 2 * Gmod * dev(eps), zero(SymmetricTensor{2, 2}))
    C = C_dil + C_dev

    return LinearElasticMaterial(C, C_dil, C_dev)
end

function plane_stress_tensors(Emod, nu)
    Gmod = Emod / (2 * (1 + nu))
    λ = Emod * nu / (1 - nu^2)
    C = gradient(eps -> λ * tr(eps) * one(eps) + 2 * Gmod * eps,
                 zero(SymmetricTensor{2, 2}))
    return LinearElasticMaterial(C, C, C)
end