using FerriteGmsh
using FerriteGmsh: Gmsh
using Ferrite
using Tensors
using LinearAlgebra
using SpecialeKursus
using WriteVTK
using Printf                      # needed for @printf in force file


function make_SENS_grid_nonotch(mesh_size::Float64, l::Float64; lc_fine::Float64 = l / 2)

    Gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 2)

    lc = mesh_size

    # --- GEOMETRY: plain square, no notch ---
    p1 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.add_point(1.0, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.add_point(1.0, 1.0, 0.0, lc)
    p4 = gmsh.model.geo.add_point(0.0, 1.0, 0.0, lc)

    l_bottom = gmsh.model.geo.add_line(p1, p2)
    l_right  = gmsh.model.geo.add_line(p2, p3)
    l_top    = gmsh.model.geo.add_line(p3, p4)
    l_left   = gmsh.model.geo.add_line(p4, p1)

cl   = gmsh.model.geo.add_curve_loop([l_bottom, l_right, l_top, l_left])
surf = gmsh.model.geo.add_plane_surface([cl])

    gmsh.model.geo.synchronize()

    # --- PHYSICAL GROUPS ---
    gmsh.model.add_physical_group(1, [l_bottom], -1, "bottom")
    gmsh.model.add_physical_group(1, [l_top],    -1, "top")
    gmsh.model.add_physical_group(1, [l_left],   -1, "left")
    gmsh.model.add_physical_group(1, [l_right],  -1, "right")
    gmsh.model.add_physical_group(2, [surf],      -1, "domain")

    # --- MESH SIZE FIELD: refine along crack line ---
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
    gmsh.model.mesh.renumber_nodes()
    gmsh.model.mesh.renumber_elements()

    grid = FerriteGmsh.togrid()
    Gmsh.finalize()

    return grid
end


function run_phasefield_option1(ES::Float64, MS::Float64;
                                l::Float64                               = 0.0150,
                                lc_fine::Float64                         = l / 2,
                                n_steps_damage::Int                      = 1000,
                                force_output_path::Union{String,Nothing} = nothing,
                                vtu_output_dir::Union{String,Nothing}    = nothing)

    # --- MESH: plain square, no geometric notch ---
    grid = make_SENS_grid_nonotch(ES, l; lc_fine = lc_fine)

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

    # --- INITIALIZE d = 1 AND H ALONG CRACK LINE ---
    # Crack runs from x = 0 to x = 0.5 at y = 0.5.
    # All nodes within lc_fine of y = 0.5 and x <= 0.5 are set to d = 1.
    # H is also set large at these locations so the phase field solver
    # cannot heal the crack during the first step.
    crack_dof_set = Set{Int}()
    crack_cell_set = Set{Int}()
    for cell in CellIterator(dh_d)
        dofs  = celldofs(cell)
        nodes = cell.nodes
        for (i, node_id) in enumerate(nodes)
            node = grid.nodes[node_id]
            x, y = node.x[1], node.x[2]
            if abs(y - 0.5) < lc_fine && x <= 0.5
                push!(crack_dof_set, dofs[i])
                push!(crack_cell_set, cellid(cell))
            end
        end
    end
    for dof in crack_dof_set
        d[dof] = 1.0
    end
    for cid in crack_cell_set
        for qp in 1:n_qp
            H[cid, qp] = 1e10
        end
    end
    println("Initialized $(length(crack_dof_set)) crack DOFs to d = 1")

    println("Mesh: $(getncells(grid)) cells, $(ndofs(dh_u)) u-dofs, $(ndofs(dh_d)) d-dofs")

    # --- OUTPUT SETUP ---
    vtu_dir = vtu_output_dir === nothing ? "." : vtu_output_dir

    if force_output_path !== nothing
        _init_force_file(force_output_path)
        bot_y_dofs = _bottom_y_dofs(dh_u, grid)
    end

    # --- VTK STEP 0 ---
    pvd = paraview_collection(joinpath(vtu_dir, "Phase-Field-Option1"))
    VTKGridFile(joinpath(vtu_dir, "phasefield_option1_step_0000"), grid) do vtk
        write_solution(vtk, dh_u, u)
        write_solution(vtk, dh_d, d)
        pvd[0.0] = vtk
    end

    t    = 0.0
    step = 0

    # --- PHASE 1: 500 elastic steps at Δu = 1e-5 mm (hardcoded) ---
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
        VTKGridFile(joinpath(vtu_dir, "phasefield_option1_step_$stepstr"), grid) do vtk
            write_solution(vtk, dh_u, u)
            write_solution(vtk, dh_d, d)
            write_cell_data(vtk, psi,    "psi")
            write_cell_data(vtk, H_cell, "H")
            pvd[t] = vtk
        end
    end

    # --- PHASE 2: damage regime, n_steps_damage steps ---
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
        VTKGridFile(joinpath(vtu_dir, "phasefield_option1_step_$stepstr"), grid) do vtk
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



Base.invokelatest(run_phasefield_option1, 0.025, 0.5;
    l                 = 0.0150,
    lc_fine           = 0.0075,
    n_steps_damage    = 1000,
    force_output_path = raw"C:\Users\Jesper\OneDrive - Danmarks Tekniske Universitet\Skrivebord\Uni\Kandidat\Speciale\Specialekursus\Phase Field\Plotting\force_disp_d_notch.txt",
    vtu_output_dir    = raw"C:\Users\Jesper\OneDrive - Danmarks Tekniske Universitet\Skrivebord\Uni\Kandidat\Speciale\Specialekursus\Phase Field\VTU\d_notch")