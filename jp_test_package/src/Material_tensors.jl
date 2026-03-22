
function plane_strain_tensors(Emod, nu)
    Gmod = Emod / (2 * (1 + nu))
    Kmod = Emod * nu / ((1 + nu) * (1 - 2 * nu))

    C_dil = gradient(eps -> Kmod * tr(eps) * one(eps), zero(SymmetricTensor{2, 2}))
    C_dev = gradient(eps -> 2 * Gmod * dev(eps), zero(SymmetricTensor{2, 2}))
    C = C_dil + C_dev

    return LinearElasticMaterial(C, C_dil, C_dev)
end