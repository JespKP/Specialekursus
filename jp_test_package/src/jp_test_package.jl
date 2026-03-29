module jp_test_package
using Ferrite
using Tensors
using LinearAlgebra


##Structs 
struct StandardIntegration{CV}
    cellvalues::CV
end

struct ReducedIntegration{CV_DIL, CV_DEV}
    cellvalues_dil::CV_DIL
    cellvalues_dev::CV_DEV
end

mutable struct PlasticState
    σ   ::SymmetricTensor{2, 2, Float64, 3}  # stress at end of last increment
    σ_y ::Float64                             # yield stress at end of last increment
    Δλ  ::Float64                             # plastic multiplier from last increment
    εp  ::Float64                             # accumulated effective plastic strain
end


struct LinearElasticMaterial
    C::SymmetricTensor{4, 2, Float64, 9}
    C_dil::SymmetricTensor{4, 2, Float64, 9}
    C_dev::SymmetricTensor{4, 2, Float64, 9}
end


#Functions
function reinit_scheme!(scheme::StandardIntegration, cell)
    reinit!(scheme.cellvalues, cell)
end

function reinit_scheme!(scheme::ReducedIntegration, cell)
    reinit!(scheme.cellvalues_dil, cell)
    reinit!(scheme.cellvalues_dev, cell)
end

function init_plastic_state(σ_y0::Float64)
    PlasticState(zero(SymmetricTensor{2, 2, Float64}), σ_y0, 0.0, 0.0)
end


get_primary_cv(scheme::StandardIntegration) = scheme.cellvalues
get_primary_cv(scheme::ReducedIntegration)  = scheme.cellvalues_dev


include("Material_tensors.jl")
include("External_traction.jl")
include("cell_assembly.jl")
include("global_assembly.jl")
include("Von_mises.jl")
include("calculate_stresses.jl")
include("run_simulation.jl")
include("yield_function.jl")
include("run_simulation_plastic.jl")
include("elastoplastic_tangent.jl")
include("Rint_assembly.jl")
include("stress_update.jl")


export PlasticState
export init_plastic_state
export assemble_external_forces!
export assemble_cell!
export assemble_global!
export plane_strain_tensors
export plane_stress_tensors
export von_mises_from_plane_strain_3d
export cell_von_mises_plane_strain
export StandardIntegration
export ReducedIntegration
export LinearElasticMaterial
export calculate_stresses
export run_simulation
export run_simulation_plastic
export run_simulation_material
export assemble_Rint!
export stress_update!
export compute_Cep

end # module JespersPackage
