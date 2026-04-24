using FerriteGmsh
using FerriteGmsh: Gmsh
using Ferrite
using Tensors
using LinearAlgebra
using SpecialeKursus
using WriteVTK
using Printf                      # needed for @printf in force file

function make_SENS_grid_option2(mesh_size::Float64, l::Float64; lc_fine::Float64 = l / 2)

    Gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 2)

    lc = mesh_size

    # --- GEOMETRY (geo kernel only) ---
    # Rectangle corners, going counter-clockwise from bottom-left
    p1 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, lc)       # bottom-left
    p2 = gmsh.model.geo.add_point(1.0, 0.0, 0.0, lc)       # bottom-right
    p3 = gmsh.model.geo.add_point(1.0, 1.0, 0.0, lc)       # top-right
    p4 = gmsh.model.geo.add_point(0.0, 1.0, 0.0, lc)       # top-left

    # Crack endpoints
    p5 = gmsh.model.geo.add_point(0.0, 0.5, 0.0, lc_fine)  # crack mouth (on left edge)
    p6 = gmsh.model.geo.add_point(0.5, 0.5, 0.0, lc_fine)  # crack tip

    # Outer boundary — left edge split at crack mouth
    l_bottom     = gmsh.model.geo.add_line(p1, p2)   # p1 → p2
    l_right      = gmsh.model.geo.add_line(p2, p3)   # p2 → p3
    l_top        = gmsh.model.geo.add_line(p3, p4)   # p3 → p4
    l_left_upper = gmsh.model.geo.add_line(p4, p5)   # p4 → p5
    l_left_lower = gmsh.model.geo.add_line(p5, p1)   # p5 → p1

    # Crack line (embedded, not part of boundary loop)
    l_crack = gmsh.model.geo.add_line(p5, p6)        # p5 → p6

    # Closed curve loop: p1→p2→p3→p4→p5→p1
    cl_outer = gmsh.model.geo.add_curve_loop([
        l_bottom,
        l_right,
        l_top,
        l_left_upper,
        l_left_lower
    ])
    surf = gmsh.model.geo.add_plane_surface([cl_outer])

    gmsh.model.geo.synchronize()

    # Embed crack line into surface so nodes are placed along it
    gmsh.model.mesh.embed(1, [l_crack], 2, surf)

    # --- PHYSICAL GROUPS ---
    gmsh.model.add_physical_group(1, [l_bottom],                    -1, "bottom")
    gmsh.model.add_physical_group(1, [l_top],                       -1, "top")
    gmsh.model.add_physical_group(1, [l_left_upper, l_left_lower],  -1, "left")
    gmsh.model.add_physical_group(1, [l_right],                     -1, "right")
    gmsh.model.add_physical_group(2, [surf],                        -1, "domain")

    # Crack physical group — required by the Crack plugin
    crack_phys = gmsh.model.add_physical_group(1, [l_crack], -1, "crack")

    # --- MESH SIZE FIELD ---
    f = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.set_number(f, "VIn",  lc_fine)
    gmsh.model.mesh.field.set_number(f, "VOut", lc)
    gmsh.model.mesh.field.set_number(f, "XMin", 0.0)
    gmsh.model.mesh.field.set_number(f, "XMax", 1.0)
    gmsh.model.mesh.field.set_number(f, "YMin", 0.5 - 0.05)
    gmsh.model.mesh.field.set_number(f, "YMax", 0.5 + 0.05)
    gmsh.model.mesh.field.set_number(f, "ZMin", -1.0)
    gmsh.model.mesh.field.set_number(f, "ZMax",  1.0)
    gmsh.model.mesh.field.set_as_background_mesh(f)

    # --- GENERATE MESH ---
    gmsh.option.set_number("Mesh.Algorithm", 8)
    gmsh.option.set_number("Mesh.RecombineAll", 1)
    gmsh.option.set_number("Mesh.SubdivisionAlgorithm", 1)
    for (dim, tag) in gmsh.model.get_entities(2)
        gmsh.model.mesh.set_recombine(dim, tag)
    end
    gmsh.model.mesh.generate(2)

    # --- DUPLICATE NODES ALONG CRACK ---
    # The Crack plugin splits nodes on the crack curve so elements on
    # each side reference separate nodes — a true displacement discontinuity.
    gmsh.plugin.setNumber("Crack", "Dimension",     1)
    gmsh.plugin.setNumber("Crack", "PhysicalGroup", crack_phys)
    gmsh.plugin.run("Crack")

    gmsh.model.mesh.renumber_nodes()
    gmsh.model.mesh.renumber_elements()

    grid = FerriteGmsh.togrid()
    Gmsh.finalize()

    return grid
end
 
function _bottom_y_dofs(dh_u, grid)
    tol          = 1e-10
    bottom_nodes = Set(i for (i, n) in enumerate(grid.nodes) if n.x[2] < tol)
    result       = Int[]
    ndofs_per_nd = ndofs_per_cell(dh_u) ÷ length(first(CellIterator(dh_u)).nodes)
    for cell in CellIterator(dh_u)
        dofs  = celldofs(cell)
        nodes = cell.nodes
        for (i, node) in enumerate(nodes)
            if node ∈ bottom_nodes
                push!(result, dofs[(i - 1) * ndofs_per_nd + 2])   # component 2 = y
            end
        end
    end
    return unique!(sort!(result))
end

function _init_force_file(path::String)
    println("Full path: ", path)
    println("Directory: ", dirname(path))
    println("Directory exists: ", isdir(dirname(path)))
    open(path, "w") do io
        println(io, "# step\tdisplacement[mm]\treaction_force[kN]")
    end
end

function _append_force(path::String, step::Int, disp::Float64, force::Float64)
    open(path, "a") do io
        @printf(io, "%d\t%.8e\t%.8e\n", step, disp, force)
    end
end

 
function run_phasefield_option2(ES::Float64, MS::Float64;
                                l::Float64                               = 0.0150,
                                lc_fine::Float64                         = l / 2,
                                n_steps_damage::Int                      = 1000,
                                force_output_path::Union{String,Nothing} = nothing,
                                vtu_output_dir::Union{String,Nothing}    = nothing)
 
    grid = make_SENS_grid_option2(ES, l; lc_fine = lc_fine)
 
    # --- FE SPACES ---
    ip_u = Lagrange{RefQuadrilateral, 1}()^2
    ip_d = Lagrange{RefQuadrilateral, 1}()
    qr   = QuadratureRule{RefQuadrilateral}(4)
 
    cellvalues_u = CellValues(qr, ip_u)
    cellvalues_d = CellValues(qr, ip_d)
 
    # --- DOF HANDLERS ---
    dh_u = DofHandler(grid)
    add!(dh_u, :u, ip_u)
    close!(dh_u)
 
    dh_d = DofHandler(grid)
    add!(dh_d, :d, ip_d)
    close!(dh_d)
 
    # --- BOUNDARY CONDITIONS ---
    u_max = 6.1e-3
    ch = ConstraintHandler(dh_u)
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> [0.0, 0.0]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "top"),    (x, t) -> t, 2))
    close!(ch)
 
    # --- MATERIALS ---
    mat_elastic  = plane_strain_tensors(210.0, 0.3)
    mat_fracture = PhaseFieldMaterial(2.7e-3, l, 1e-7)
 
    # --- HISTORY FIELD ---
    n_qp = getnquadpoints(cellvalues_u)
    H    = zeros(getncells(grid), n_qp)
 
    # --- PRE-ALLOCATE ---
    K_u = allocate_matrix(dh_u)
    K_d = allocate_matrix(dh_d)
    f_u = zeros(ndofs(dh_u))
    f_d = zeros(ndofs(dh_d))
    u   = zeros(ndofs(dh_u))
    d   = zeros(ndofs(dh_d))
 
    println("Mesh: $(getncells(grid)) cells, $(ndofs(dh_u)) u-dofs, $(ndofs(dh_d)) d-dofs")
 
    # --- OUTPUT SETUP ---
    vtu_dir = vtu_output_dir === nothing ? "." : vtu_output_dir
 
    if force_output_path !== nothing
        _init_force_file(force_output_path)
        bot_y_dofs = _bottom_y_dofs(dh_u, grid)
    end
 
    # --- VTK STEP 0 ---
    pvd = paraview_collection(joinpath(vtu_dir, "Phase-Field-Option2"))
    VTKGridFile(joinpath(vtu_dir, "phasefield_option2_step_0000"), grid) do vtk
        write_solution(vtk, dh_u, u)
        write_solution(vtk, dh_d, d)
        pvd[0.0] = vtk
    end
 
    t    = 0.0
    step = 0
 
    # --- PHASE 1: 500 elastic steps at Δu = 1e-5 mm ---
    for _ in 1:500
        step += 1
        t    += 1e-5
        Ferrite.update!(ch, t)
        println("Phase 1 | Step $step / $(500 + n_steps_damage),  t = $t,  max(d) = $(maximum(d))")
 
        update_history!(H, cellvalues_u, dh_u, u, mat_elastic)
        solve_phasefield!(K_d, f_d, cellvalues_d, dh_d, d, H, mat_fracture)
        f_int = solve_mechanics!(K_u, f_u, u, cellvalues_u, cellvalues_d,
                                 dh_u, dh_d, d, mat_elastic, mat_fracture, ch)
 
        if force_output_path !== nothing
            rf = abs(sum(f_int[dof] for dof in bot_y_dofs))
            _append_force(force_output_path, step, t, rf)
        end
 
        psi     = compute_cell_psi(cellvalues_u, cellvalues_d, dh_u, dh_d, u, d, mat_elastic)
        H_cell  = compute_cell_H(H, cellvalues_u, dh_u)
        stepstr = lpad(step, 4, '0')
        VTKGridFile(joinpath(vtu_dir, "phasefield_option2_step_$stepstr"), grid) do vtk
            write_solution(vtk, dh_u, u)
            write_solution(vtk, dh_d, d)
            write_cell_data(vtk, psi,    "psi")
            write_cell_data(vtk, H_cell, "H")
            pvd[t] = vtk
        end
    end
 
    # --- PHASE 2: damage regime ---
    Δu₂ = (u_max - t) / n_steps_damage
 
    for _ in 1:n_steps_damage
        step += 1
        t     = min(t + Δu₂, u_max)
        Ferrite.update!(ch, t)
        println("Phase 2 | Step $step / $(500 + n_steps_damage),  t = $t,  max(d) = $(maximum(d))")
 
        update_history!(H, cellvalues_u, dh_u, u, mat_elastic)
        solve_phasefield!(K_d, f_d, cellvalues_d, dh_d, d, H, mat_fracture)
        f_int = solve_mechanics!(K_u, f_u, u, cellvalues_u, cellvalues_d,
                                 dh_u, dh_d, d, mat_elastic, mat_fracture, ch)
 
        if force_output_path !== nothing
            rf = abs(sum(f_int[dof] for dof in bot_y_dofs))
            _append_force(force_output_path, step, t, rf)
        end
 
        psi     = compute_cell_psi(cellvalues_u, cellvalues_d, dh_u, dh_d, u, d, mat_elastic)
        H_cell  = compute_cell_H(H, cellvalues_u, dh_u)
        stepstr = lpad(step, 4, '0')
        VTKGridFile(joinpath(vtu_dir, "phasefield_option2_step_$stepstr"), grid) do vtk
            write_solution(vtk, dh_u, u)
            write_solution(vtk, dh_d, d)
            write_cell_data(vtk, psi,    "psi")
            write_cell_data(vtk, H_cell, "H")
            pvd[t] = vtk
        end
    end
 
    vtk_save(pvd)
    return u, d, H, dh_u, dh_d, grid
end


Base.invokelatest(run_phasefield_option2, 0.025, 0.5;
    l                 = 0.0150,
    lc_fine           = 0.0075,
    n_steps_damage    = 1000,
    force_output_path = raw"C:\Users\Jesper\OneDrive - Danmarks Tekniske Universitet\Skrivebord\Uni\Kandidat\Speciale\Specialekursus\Phase Field\Plotting\force_disp_thin_crack.txt",
    vtu_output_dir    = raw"C:\Users\Jesper\OneDrive - Danmarks Tekniske Universitet\Skrivebord\Uni\Kandidat\Speciale\Specialekursus\Phase Field\VTU\thin_crack")