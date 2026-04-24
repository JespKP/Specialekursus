using FerriteGmsh
using FerriteGmsh: Gmsh
using Ferrite
using Tensors
using LinearAlgebra
using SpecialeKursus
using WriteVTK
using Printf                      # needed for @printf in force file

function make_SENS_grid(mesh_size::Float64, notch_length::Float64, notch_width::Float64, l::Float64; lc_fine::Float64 = l / 2)

    Gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 2)

    lc      = mesh_size
    lc_fine = lc_fine

    # --- GEOMETRY ---
    rect  = gmsh.model.occ.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    notch = gmsh.model.occ.add_rectangle(
        0.0,
        0.5 - notch_width / 2,
        0.0,
        notch_length,
        notch_width
    )

    gmsh.model.occ.cut([(2, rect)], [(2, notch)])
    gmsh.model.occ.synchronize()

    # --- PHYSICAL GROUPS ---
    eps = 1e-6
    bottom_curves = gmsh.model.get_entities_in_bounding_box(-eps, -eps,  -eps, 1+eps, eps,   eps, 1)
    top_curves    = gmsh.model.get_entities_in_bounding_box(-eps, 1-eps, -eps, 1+eps, 1+eps, eps, 1)
    left_curves   = gmsh.model.get_entities_in_bounding_box(-eps, -eps,  -eps, eps,   1+eps, eps, 1)
    right_curves  = gmsh.model.get_entities_in_bounding_box(1-eps, -eps, -eps, 1+eps, 1+eps, eps, 1)

    gmsh.model.add_physical_group(1, [c[2] for c in bottom_curves], -1, "bottom")
    gmsh.model.add_physical_group(1, [c[2] for c in top_curves],    -1, "top")
    gmsh.model.add_physical_group(1, [c[2] for c in left_curves],   -1, "left")
    gmsh.model.add_physical_group(1, [c[2] for c in right_curves],  -1, "right")

    surfaces = gmsh.model.get_entities(2)
    gmsh.model.add_physical_group(2, [s[2] for s in surfaces], -1, "domain")

    # --- MESH SIZE FIELD: refine vertical strip from notch tip upward ---
    f = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.set_number(f, "VIn",    lc_fine)
    gmsh.model.mesh.field.set_number(f, "VOut",   lc)
    gmsh.model.mesh.field.set_number(f, "XMin",   0.5)
    gmsh.model.mesh.field.set_number(f, "XMax",   1.0)
    gmsh.model.mesh.field.set_number(f, "YMin",   0.5 - 0.05)
    gmsh.model.mesh.field.set_number(f, "YMax",   0.5 + 0.05)
    gmsh.model.mesh.field.set_number(f, "ZMin",  -1.0)
    gmsh.model.mesh.field.set_number(f, "ZMax",   1.0)
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


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers (reaction force file)
# ─────────────────────────────────────────────────────────────────────────────

"""
    _bottom_y_dofs(dh_u, grid)

Returns the unique global DOF indices for the y-displacement at all nodes on
the bottom edge (y ≈ 0).  Called once before the load-stepping loop.
"""
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


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_phasefield(ES, MS;
                   force_output_path = nothing,
                   vtu_output_dir    = nothing)

Runs the staggered phase-field simulation.

Keyword arguments
-----------------
- `force_output_path`  Path to the reaction-force text file
                       e.g. `"output/force_disp.txt"`.
                       One tab-separated row per step: step, displacement, force.
                       Pass `nothing` to skip.

- `vtu_output_dir`     Directory for VTU files
                       e.g. `"output/vtu"`.
                       Each step writes `phasefield_SENS_step_XXXX.vtu`
                       containing u, d, and psi (strain energy density).
                       The paraview collection is also placed here.
                       Pass `nothing` to keep the original behaviour
                       (writes to the current working directory).

Returns
-------
`(u, d, H, dh_u, dh_d, grid)`
"""
function run_phasefield(ES::Float64, MS::Float64;
                        l::Float64                               = 0.0150,
                        lc_fine::Float64                         = l / 2,
                        n_steps_damage::Int                      = 1000,
                        force_output_path::Union{String,Nothing} = nothing,
                        vtu_output_dir::Union{String,Nothing}    = nothing)

    # --- MESH ---

    grid = make_SENS_grid(ES, MS, 0.005, l, lc_fine=lc_fine)
   
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

bottom_corner = Set(i for (i, n) in enumerate(grid.nodes)
                    if n.x[2] < 1e-10 && n.x[1] < 1e-10)
addnodeset!(grid, "bottom_corner", bottom_corner)

add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"),         (x, t) -> 0.0, 2))  # y locked on all bottom
add!(ch, Dirichlet(:u, getnodeset(grid, "bottom_corner"),   (x, t) -> 0.0, 1))  # x locked at corner only
add!(ch, Dirichlet(:u, getfacetset(grid, "top"),            (x, t) -> t,   2))  # prescribed displacement on top
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
    pvd = paraview_collection(joinpath(vtu_dir, "Phase-Field"))
    VTKGridFile(joinpath(vtu_dir, "phasefield_SENS_step_0000"), grid) do vtk
        write_solution(vtk, dh_u, u)
        write_solution(vtk, dh_d, d)
        pvd[0.0] = vtk
    end

    t    = 0.0
step = 0

u_peak   = 5.8e-3    # load to just before fracture
u_valley = 0.0       # unload back to zero
u_max    = 6.1e-3    # final reload target

Δu₁ = 1e-5                                    # elastic phase (hardcoded 500 steps)
Δu₂ = (u_peak   - 5e-3) / n_steps_damage      # loading phase
Δu₃ = (u_peak   - u_valley) / n_steps_damage  # unloading phase
Δu₄ = (u_max    - u_valley) / n_steps_damage  # reloading phase

# --- PHASE 1: 500 elastic steps at Δu = 1e-5 mm ---
for _ in 1:500
    step += 1
    t    += Δu₁
    Ferrite.update!(ch, t)
    println("Phase 1 | Step $step,  t = $t,  max(d) = $(maximum(d))")

    update_history!(H, cellvalues_u, dh_u, u, mat_elastic)
    solve_phasefield!(K_d, f_d, cellvalues_d, dh_d, d, H, mat_fracture)
    f_int = solve_mechanics!(K_u, f_u, u, cellvalues_u, cellvalues_d,
                             dh_u, dh_d, d, mat_elastic, mat_fracture, ch)

    if force_output_path !== nothing
        rf = abs(sum(f_int[dof] for dof in bot_y_dofs))
        _append_force(force_output_path, step, t, rf)
    end

    psi    = compute_cell_psi(cellvalues_u, cellvalues_d, dh_u, dh_d, u, d, mat_elastic)
    H_cell = compute_cell_H(H, cellvalues_u, dh_u)
    stepstr = lpad(step, 4, '0')
    VTKGridFile(joinpath(vtu_dir, "phasefield_SENS_step_$stepstr"), grid) do vtk
        write_solution(vtk, dh_u, u)
        write_solution(vtk, dh_d, d)
        write_cell_data(vtk, psi,    "psi")
        write_cell_data(vtk, H_cell, "H")
        pvd[t] = vtk
    end
end

# --- PHASE 2: loading to u_peak ---
for _ in 1:n_steps_damage
    step += 1
    t     = min(t + Δu₂, u_peak)
    Ferrite.update!(ch, t)
    println("Phase 2 (loading) | Step $step,  t = $t,  max(d) = $(maximum(d))")

    update_history!(H, cellvalues_u, dh_u, u, mat_elastic)
    solve_phasefield!(K_d, f_d, cellvalues_d, dh_d, d, H, mat_fracture)
    f_int = solve_mechanics!(K_u, f_u, u, cellvalues_u, cellvalues_d,
                             dh_u, dh_d, d, mat_elastic, mat_fracture, ch)

    if force_output_path !== nothing
        rf = abs(sum(f_int[dof] for dof in bot_y_dofs))
        _append_force(force_output_path, step, t, rf)
    end

    psi    = compute_cell_psi(cellvalues_u, cellvalues_d, dh_u, dh_d, u, d, mat_elastic)
    H_cell = compute_cell_H(H, cellvalues_u, dh_u)
    stepstr = lpad(step, 4, '0')
    VTKGridFile(joinpath(vtu_dir, "phasefield_SENS_step_$stepstr"), grid) do vtk
        write_solution(vtk, dh_u, u)
        write_solution(vtk, dh_d, d)
        write_cell_data(vtk, psi,    "psi")
        write_cell_data(vtk, H_cell, "H")
        pvd[t] = vtk
    end
end

# --- PHASE 3: unloading to u_valley ---
for _ in 1:n_steps_damage
    step += 1
    t     = max(t - Δu₃, u_valley)
    Ferrite.update!(ch, t)
    println("Phase 3 (unloading) | Step $step,  t = $t,  max(d) = $(maximum(d))")

    update_history!(H, cellvalues_u, dh_u, u, mat_elastic)
    solve_phasefield!(K_d, f_d, cellvalues_d, dh_d, d, H, mat_fracture)
    f_int = solve_mechanics!(K_u, f_u, u, cellvalues_u, cellvalues_d,
                             dh_u, dh_d, d, mat_elastic, mat_fracture, ch)

    if force_output_path !== nothing
        rf = abs(sum(f_int[dof] for dof in bot_y_dofs))
        _append_force(force_output_path, step, t, rf)
    end

    psi    = compute_cell_psi(cellvalues_u, cellvalues_d, dh_u, dh_d, u, d, mat_elastic)
    H_cell = compute_cell_H(H, cellvalues_u, dh_u)
    stepstr = lpad(step, 4, '0')
    VTKGridFile(joinpath(vtu_dir, "phasefield_SENS_step_$stepstr"), grid) do vtk
        write_solution(vtk, dh_u, u)
        write_solution(vtk, dh_d, d)
        write_cell_data(vtk, psi,    "psi")
        write_cell_data(vtk, H_cell, "H")
        pvd[t] = vtk
    end
end

# --- PHASE 4: reloading to u_max ---
for _ in 1:n_steps_damage
    step += 1
    t     = min(t + Δu₄, u_max)
    Ferrite.update!(ch, t)
    println("Phase 4 (reloading) | Step $step,  t = $t,  max(d) = $(maximum(d))")

    update_history!(H, cellvalues_u, dh_u, u, mat_elastic)
    solve_phasefield!(K_d, f_d, cellvalues_d, dh_d, d, H, mat_fracture)
    f_int = solve_mechanics!(K_u, f_u, u, cellvalues_u, cellvalues_d,
                             dh_u, dh_d, d, mat_elastic, mat_fracture, ch)

    if force_output_path !== nothing
        rf = abs(sum(f_int[dof] for dof in bot_y_dofs))
        _append_force(force_output_path, step, t, rf)
    end

    psi    = compute_cell_psi(cellvalues_u, cellvalues_d, dh_u, dh_d, u, d, mat_elastic)
    H_cell = compute_cell_H(H, cellvalues_u, dh_u)
    stepstr = lpad(step, 4, '0')
    VTKGridFile(joinpath(vtu_dir, "phasefield_SENS_step_$stepstr"), grid) do vtk
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

# Medium — h = l/2 (paper recommendation, expected to be sufficient)
Base.invokelatest(run_phasefield, 0.025, 0.5;
    l                 = 0.0150,
    lc_fine           = 0.0075,
    n_steps_damage    = 1000,
    force_output_path = raw"C:\Users\Jesper\OneDrive - Danmarks Tekniske Universitet\Skrivebord\Uni\Kandidat\Speciale\Specialekursus\Phase Field\Plotting\force_disp_LoadUnload.txt",
    vtu_output_dir    = raw"C:\Users\Jesper\OneDrive - Danmarks Tekniske Universitet\Skrivebord\Uni\Kandidat\Speciale\Specialekursus\Phase Field\VTU\LoadUnload")