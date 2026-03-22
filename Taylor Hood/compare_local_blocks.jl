using Ferrite

using Tensors
using LinearAlgebra

function create_cook_grid(nx, ny)
    corners = [
        Vec{2}((0.0, 0.0)),
        Vec{2}((48.0, 44.0)),
        Vec{2}((48.0, 60.0)),
        Vec{2}((0.0, 44.0)),
    ]
    return generate_grid(Triangle, (nx, ny), corners)
end

function create_values(ip_u, ip_p)
    qr = QuadratureRule{RefTriangle}(3)
    cv_u = CellValues(qr, ip_u)
    cv_p = CellValues(qr, ip_p)
    return cv_u, cv_p
end

function dev_3d(t::SymmetricTensor{2, 2, T}) where {T}
    return dev(SymmetricTensor{2, 3}((i, j) -> (i <= 2 && j <= 2) ? t[i, j] : zero(T)))
end

# Reference from Exampkle.jl
function ke_reference!(ke, cv_u, cv_p, Gmod, Kmod)
    nu = getnbasefunctions(cv_u)
    np = getnbasefunctions(cv_p)
    ur = 1:nu
    pr = (nu + 1):(nu + np)

    fill!(ke, 0.0)

    for qp in 1:getnquadpoints(cv_u)
        dΩ = getdetJdV(cv_u, qp)

        for i in 1:nu
            eps_i = dev_3d(symmetric(shape_gradient(cv_u, qp, i)))
            for j in 1:i
                eps_j = dev_3d(symmetric(shape_gradient(cv_u, qp, j)))
                ke[ur[i], ur[j]] += 2 * Gmod * (eps_i ⊡ eps_j) * dΩ
            end
        end

        for i in 1:np
            Ni = shape_value(cv_p, qp, i)
            for j in 1:nu
                divNj = shape_divergence(cv_u, qp, j)
                ke[pr[i], ur[j]] += -Ni * divNj * dΩ
            end
            for j in 1:i
                Nj = shape_value(cv_p, qp, j)
                ke[pr[i], pr[j]] += -(1 / Kmod) * Ni * Nj * dΩ
            end
        end
    end

    # Symmetrize lower triangle as in Exampkle
    for i in 1:size(ke, 1)
        for j in (i + 1):size(ke, 2)
            ke[i, j] = ke[j, i]
        end
    end

    return ke
end

# User formulation from TaylorHood.jl
function ke_user!(ke, cv_u, cv_p, Gmod, Kmod)
    nu = getnbasefunctions(cv_u)
    np = getnbasefunctions(cv_p)
    ur = 1:nu
    pr = (nu + 1):(nu + np)

    fill!(ke, 0.0)

    for qp in 1:getnquadpoints(cv_u)
        dΩ = getdetJdV(cv_u, qp)

        for i in 1:nu
            gradNi = shape_symmetric_gradient(cv_u, qp, i)
            for j in 1:np
                Nj = shape_value(cv_p, qp, j)
                ke[ur[i], pr[j]] += -Nj * tr(gradNi) * dΩ
            end
        end
    end

    for qp in 1:getnquadpoints(cv_u)
        dΩ = getdetJdV(cv_u, qp)
        for j in 1:nu
            gradNj = shape_symmetric_gradient(cv_u, qp, j)
            for i in 1:nu
                gradNi = shape_symmetric_gradient(cv_u, qp, i)
                devNi = dev(gradNi)
                ke[ur[j], ur[i]] += 2 * Gmod * (devNi ⊡ gradNj) * dΩ
            end
        end
    end

    for qp in 1:getnquadpoints(cv_u)
        dΩ = getdetJdV(cv_u, qp)
        for i in 1:np
            Ni = shape_value(cv_p, qp, i)
            for j in 1:nu
                gradNj = shape_symmetric_gradient(cv_u, qp, j)
                ke[pr[i], ur[j]] += -Ni * tr(gradNj) * dΩ
            end
        end
    end

    for qp in 1:getnquadpoints(cv_u)
        dΩ = getdetJdV(cv_u, qp)
        for j in 1:np
            Nj = shape_value(cv_p, qp, j)
            for i in 1:np
                Ni = shape_value(cv_p, qp, i)
                ke[pr[j], pr[i]] += -(1 / Kmod) * Nj * Ni * dΩ
            end
        end
    end

    return ke
end

function block_view(ke, nu, np)
    ur = 1:nu
    pr = (nu + 1):(nu + np)
    return ke[ur, ur], ke[ur, pr], ke[pr, ur], ke[pr, pr]
end

function compare_local_blocks(; Emod = 1.0, nu = 0.5, nx = 2, ny = 2, cellid_to_check = 1)
    ip_u = Lagrange{RefTriangle, 2}()^2
    ip_p = Lagrange{RefTriangle, 1}()
    cv_u, cv_p = create_values(ip_u, ip_p)

    grid = create_cook_grid(nx, ny)
    dh = DofHandler(grid)
    add!(dh, :u, ip_u)
    add!(dh, :p, ip_p)
    close!(dh)

    Gmod = Emod / (2 * (1 + nu))
    Kmod = Emod * nu / (3 * (1 - 2 * nu))

    nu_b = getnbasefunctions(cv_u)
    np_b = getnbasefunctions(cv_p)
    nloc = nu_b + np_b

    ke_ref = zeros(nloc, nloc)
    ke_usr = zeros(nloc, nloc)

    for cell in CellIterator(dh)
        if Ferrite.cellid(cell) == cellid_to_check
            reinit!(cv_u, cell)
            reinit!(cv_p, cell)
            ke_reference!(ke_ref, cv_u, cv_p, Gmod, Kmod)
            ke_user!(ke_usr, cv_u, cv_p, Gmod, Kmod)
            break
        end
    end

    Kuu_ref, Kup_ref, Kpu_ref, Kpp_ref = block_view(ke_ref, nu_b, np_b)
    Kuu_usr, Kup_usr, Kpu_usr, Kpp_usr = block_view(ke_usr, nu_b, np_b)

    println("Comparison setup:")
    println("  cell id = ", cellid_to_check)
    println("  E = ", Emod, ", nu = ", nu)
    println("  n_u = ", nu_b, ", n_p = ", np_b)
    println("  Coordinates of this cell:")
    println("  ", getcoordinates(grid, cellid_to_check))

    println("\nAbsolute Frobenius norm of block differences ||K_user - K_ref||:")
    println("  uu: ", norm(Kuu_usr - Kuu_ref))
    println("  up: ", norm(Kup_usr - Kup_ref))
    println("  pu: ", norm(Kpu_usr - Kpu_ref))
    println("  pp: ", norm(Kpp_usr - Kpp_ref))

    println("\nReference symmetry checks:")
    println("  ||Kuu - Kuu' || = ", norm(Kuu_ref - Kuu_ref'))
    println("  ||Kup - Kpu' || = ", norm(Kup_ref - Kpu_ref'))
    println("  ||Kpp - Kpp' || = ", norm(Kpp_ref - Kpp_ref'))

    println("\nUser symmetry checks:")
    println("  ||Kuu - Kuu' || = ", norm(Kuu_usr - Kuu_usr'))
    println("  ||Kup - Kpu' || = ", norm(Kup_usr - Kpu_usr'))
    println("  ||Kpp - Kpp' || = ", norm(Kpp_usr - Kpp_usr'))
end

compare_local_blocks()
