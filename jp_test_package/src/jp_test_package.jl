module jp_test_package
using Ferrite
using Tensors


##Structs 
struct StandardIntegration{CV}
    cellvalues::CV
end

struct ReducedIntegration{CV_DIL, CV_DEV}
    cellvalues_dil::CV_DIL
    cellvalues_dev::CV_DEV
end


struct LinearElasticMaterial
    C::SymmetricTensor{4, 2, Float64}
    C_dil::SymmetricTensor{4, 2, Float64}
    C_dev::SymmetricTensor{4, 2, Float64}
end


#Functions
function reinit_scheme!(scheme::StandardIntegration, cell)
    reinit!(scheme.cellvalues, cell)
end

function reinit_scheme!(scheme::ReducedIntegration, cell)
    reinit!(scheme.cellvalues_dil, cell)
    reinit!(scheme.cellvalues_dev, cell)
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

export assemble_external_forces!
export assemble_cell!
export assemble_global!
export plane_strain_tensors
export von_mises_from_plane_strain_3d
export cell_von_mises_plane_strain
export StandardIntegration
export ReducedIntegration
export LinearElasticMaterial
export calculate_stresses
export run_simulation

end # module JespersPackage
