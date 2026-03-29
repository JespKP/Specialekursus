function assemble_cell!(ke, scheme::ReducedIntegration, mat::LinearElasticMaterial)

    for q_point in 1:getnquadpoints(scheme.cellvalues_dil)
        dΩ = getdetJdV(scheme.cellvalues_dil, q_point)
        for i in 1:getnbasefunctions(scheme.cellvalues_dil)
            ∇Nᵢ = shape_gradient(scheme.cellvalues_dil, q_point, i)
            ∇ˢʸᵐNᵢ = symmetric(∇Nᵢ)
            for j in 1:getnbasefunctions(scheme.cellvalues_dil)
                ∇Nⱼ = shape_gradient(scheme.cellvalues_dil, q_point, j)
                ∇ˢʸᵐNⱼ = symmetric(∇Nⱼ)
                ke[i, j] += (∇ˢʸᵐNᵢ ⊡ mat.C_dil ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
     for q_point in 1:getnquadpoints(scheme.cellvalues_dev)
        dΩ = getdetJdV(scheme.cellvalues_dev, q_point)
        for i in 1:getnbasefunctions(scheme.cellvalues_dev)
            ∇Nᵢ = shape_gradient(scheme.cellvalues_dev, q_point, i)
            ∇ˢʸᵐNᵢ = symmetric(∇Nᵢ)
            for j in 1:getnbasefunctions(scheme.cellvalues_dev)
                ∇Nⱼ = shape_gradient(scheme.cellvalues_dev, q_point, j)
                ∇ˢʸᵐNⱼ = symmetric(∇Nⱼ)
                ke[i, j] += (∇ˢʸᵐNᵢ ⊡ mat.C_dev ⊡ ∇ˢʸᵐNⱼ) * dΩ
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
            ∇ˢʸᵐNᵢ = symmetric(∇Nᵢ)
            for j in 1:getnbasefunctions(scheme.cellvalues)
                # Symmetric gradient of the trial function
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(scheme.cellvalues, q_point, j)
                ke[i, j] += (∇ˢʸᵐNᵢ ⊡ mat.C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end

function assemble_cell!(ke, scheme::StandardIntegration, mat::LinearElasticMaterial,
                        committed_cell_states::Vector{PlasticState},
                        trial_cell_states::Vector{PlasticState}, h::Float64)
    for q_point in 1:getnquadpoints(scheme.cellvalues)
        dΩ             = getdetJdV(scheme.cellvalues, q_point)
        committed_state = committed_cell_states[q_point]
        trial_state     = trial_cell_states[q_point]
        # tangent at σ_committed, consistent with explicit stress update
        C_tangent = trial_state.Δλ > 0 ? compute_Cep(committed_state.σ, mat, h) : mat.C
        for i in 1:getnbasefunctions(scheme.cellvalues)
            ∇ˢʸᵐNᵢ = shape_symmetric_gradient(scheme.cellvalues, q_point, i)
            for j in 1:getnbasefunctions(scheme.cellvalues)
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(scheme.cellvalues, q_point, j)
                ke[i, j] += (∇ˢʸᵐNᵢ ⊡ C_tangent ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end