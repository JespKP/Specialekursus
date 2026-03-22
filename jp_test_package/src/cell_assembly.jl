function assemble_cell!(ke, scheme::ReducedIntegration, mat::LinearElasticMaterial)

    for q_point in 1:getnquadpoints(scheme.cellvalues_dil)
        dΩ = getdetJdV(scheme.cellvalues_dil, q_point)
        for i in 1:getnbasefunctions(scheme.cellvalues_dil)
            ∇Nᵢ = shape_gradient(scheme.cellvalues_dil, q_point, i)
            for j in 1:getnbasefunctions(scheme.cellvalues_dil)
                ∇Nⱼ = shape_gradient(scheme.cellvalues_dil, q_point, j)
                ∇ˢʸᵐNⱼ = symmetric(∇Nⱼ)
                ke[i, j] += (∇Nᵢ ⊡ mat.C_dil ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
     for q_point in 1:getnquadpoints(scheme.cellvalues_dev)
        dΩ = getdetJdV(scheme.cellvalues_dev, q_point)
        for i in 1:getnbasefunctions(scheme.cellvalues_dev)
            ∇Nᵢ = shape_gradient(scheme.cellvalues_dev, q_point, i)
            for j in 1:getnbasefunctions(scheme.cellvalues_dev)
                ∇Nⱼ = shape_gradient(scheme.cellvalues_dev, q_point, j)
                ∇ˢʸᵐNⱼ = symmetric(∇Nⱼ)
                ke[i, j] += (∇Nᵢ ⊡ mat.C_dev ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end

function assemble_cell!(ke, scheme::StandardIntegration, mat::LinearElasticMaterial)

    for q_point in 1:getnquadpoints(scheme.cellvalues)
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(scheme.cellvalues, q_point)
        for i in 1:getnbasefunctions(scheme.cellvalues)
            # Gradient of the test function
            ∇Nᵢ = shape_gradient(scheme.cellvalues, q_point, i)
            for j in 1:getnbasefunctions(scheme.cellvalues)
                # Symmetric gradient of the trial function
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(scheme.cellvalues, q_point, j)
                ke[i, j] += (∇Nᵢ ⊡ mat.C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end