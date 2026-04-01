"""
    von_mises_from_plane_strain_3d(eps3, C2; return_components=true)

Plane-strain convenience overload that accepts a 3D strain tensor `eps3`
and a 2D constitutive tensor `C2` (as returned by `plane_strain_tensors`).
The out-of-plane normal stress is reconstructed as `sigma33 = lambda * (eps11 + eps22)`.
"""
function von_mises_from_plane_strain_3d(
    eps3::SymmetricTensor{2, 3, T1},
    C2::SymmetricTensor{4, 2, T2};
    return_components=true,
) where {T1, T2}
    # Plane-strain 2D tensor as (σ_11, σ_12=σ_21, σ_22)
    eps2 = SymmetricTensor{2, 2}((eps3[1, 1], eps3[2, 2], eps3[1, 2]))
    sigma2 = C2 ⊡ eps2

    sigma11 = sigma2[1, 1]
    sigma22 = sigma2[2, 2]
    sigma12 = sigma2[1, 2]

    # For isotropic plane strain, lambda equals C_1122 in the 2D constitutive tensor.
    lambda = C2[1, 1, 2, 2]
    sigma33 = lambda * (eps3[1, 1] + eps3[2, 2])

    sigma13 = zero(promote_type(T1, T2))
    sigma23 = zero(promote_type(T1, T2))

    sigma_vm = sqrt(
        0.5 * ((sigma11 - sigma22)^2 + (sigma22 - sigma33)^2 + (sigma33 - sigma11)^2) +
        3.0 * (sigma12^2 + sigma13^2 + sigma23^2),
    )

    if return_components
        return (
            sigma_vm = sigma_vm,
            sigma11 = sigma11,
            sigma22 = sigma22,
            sigma33 = sigma33,
            sigma12 = sigma12,
            sigma13 = sigma13,
            sigma23 = sigma23,
        )
    end

    return sigma_vm
end

"""
    von_mises_from_plane_strain_3d(eps3, C3; return_components=true)

Plane-strain convenience overload that accepts a 3D strain tensor `eps3`
and a 3D constitutive tensor `C3`.
"""
function von_mises_from_plane_strain_3d(
    eps3::SymmetricTensor{2, 3, T1},
    C3::SymmetricTensor{4, 3, T2};
    return_components=true,
) where {T1, T2}
    sigma3 = C3 ⊡ eps3

    sigma11 = sigma3[1, 1]
    sigma22 = sigma3[2, 2]
    sigma33 = sigma3[3, 3]
    sigma12 = sigma3[1, 2]
    sigma13 = sigma3[1, 3]
    sigma23 = sigma3[2, 3]

    sigma_vm = sqrt(
        0.5 * ((sigma11 - sigma22)^2 + (sigma22 - sigma33)^2 + (sigma33 - sigma11)^2) +
        3.0 * (sigma12^2 + sigma13^2 + sigma23^2),
    )

    if return_components
        return (
            sigma_vm = sigma_vm,
            sigma11 = sigma11,
            sigma22 = sigma22,
            sigma33 = sigma33,
            sigma12 = sigma12,
            sigma13 = sigma13,
            sigma23 = sigma23,
        )
    end

    return sigma_vm
end

"""
    cell_von_mises_plane_strain(grid, dh, cellvalues_dev, u, C)

Compute area-averaged von Mises stress per cell for a 2D plane-strain problem.
Returns a vector with one value per cell.
"""



function cell_von_mises_plane_strain(grid, dh, scheme, u, mat::LinearElasticMaterial)
    cell_vm = zeros(getncells(grid))

    for cell in CellIterator(dh)
        reinit_scheme!(scheme, cell)
        area = 0.0
        vm_int = 0.0

        for qp in 1:getnquadpoints(get_primary_cv(scheme))
            eps = function_symmetric_gradient(get_primary_cv(scheme), qp, u, celldofs(cell))
            eps3 = SymmetricTensor{2, 3}((eps[1, 1], eps[2, 2], 0.0, eps[1, 2], 0.0, 0.0))
            sigma_vm = von_mises_from_plane_strain_3d(eps3, mat.C_dev; return_components=false)

            dOmega = getdetJdV(get_primary_cv(scheme), qp)
            vm_int += sigma_vm * dOmega
            area += dOmega
        end

        cell_vm[cellid(cell)] = vm_int / area
    end

    return cell_vm
end
