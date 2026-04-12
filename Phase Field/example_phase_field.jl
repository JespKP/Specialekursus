using FerriteGmsh
using FerriteGmsh: Gmsh
using Ferrite
using Tensors
using LinearAlgebra
using SpecialeKursus

function make_SENS_grid(mesh_size::Float64, notch_length::Float64, notch_width::Float64)

    Gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 2)

    lc       = mesh_size
    lc_fine  = mesh_size / 3    # finer near notch tip

    # --- MAIN SQUARE: 1x1 ---
    rect = gmsh.model.occ.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)

    # --- NOTCH: thin rectangle cut from left edge to centre ---
    # Centred at y=0.5, extends from x=0 to x=notch_length
    # notch_width is very small, e.g. 0.005
    notch = gmsh.model.occ.add_rectangle(
        0.0,                        # x start — left edge
        0.5 - notch_width / 2,      # y start — just below mid
        0.0,
        notch_length,               # x length
        notch_width                 # y height
    )

    # --- CUT the notch out of the square ---
    gmsh.model.occ.cut([(2, rect)], [(2, notch)])
    gmsh.model.occ.synchronize()

    # --- PHYSICAL GROUPS ---
    # Get the boundary curves after the cut
    # Use bounding box to identify each boundary
    eps = 1e-6

    bottom_curves = gmsh.model.get_entities_in_bounding_box(
        -eps, -eps, -eps, 1+eps, eps, eps, 1)
    top_curves = gmsh.model.get_entities_in_bounding_box(
        -eps, 1-eps, -eps, 1+eps, 1+eps, eps, 1)
    left_curves = gmsh.model.get_entities_in_bounding_box(
        -eps, -eps, -eps, eps, 1+eps, eps, 1)
    right_curves = gmsh.model.get_entities_in_bounding_box(
        1-eps, -eps, -eps, 1+eps, 1+eps, eps, 1)

    gmsh.model.add_physical_group(1, [c[2] for c in bottom_curves], -1, "bottom")
    gmsh.model.add_physical_group(1, [c[2] for c in top_curves],    -1, "top")
    gmsh.model.add_physical_group(1, [c[2] for c in left_curves],   -1, "left")
    gmsh.model.add_physical_group(1, [c[2] for c in right_curves],  -1, "right")

    # Mark the whole domain as physical surface
    surfaces = gmsh.model.get_entities(2)
    gmsh.model.add_physical_group(2, [s[2] for s in surfaces], -1, "domain")

    # --- MESH SIZE FIELD: refine near notch tip ---
    f = gmsh.model.mesh.field.add("Ball")
    gmsh.model.mesh.field.set_number(f, "Radius",   notch_width * 10)
    gmsh.model.mesh.field.set_number(f, "VIn",      lc_fine)
    gmsh.model.mesh.field.set_number(f, "VOut",     lc)
    gmsh.model.mesh.field.set_number(f, "XCenter",  notch_length)
    gmsh.model.mesh.field.set_number(f, "YCenter",  0.5)
    gmsh.model.mesh.field.set_number(f, "ZCenter",  0.0)
    gmsh.model.mesh.field.set_as_background_mesh(f)

    # --- GENERATE MESH ---
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.renumber_nodes()
    gmsh.model.mesh.renumber_elements()

    grid = FerriteGmsh.togrid()
    Gmsh.finalize()

    return grid
end



function run_phasefield(nx::Int, ny::Int)

    # --- MESH ---
    grid = make_SENS_grid(0.01, 0.5, 0.005)

    # --- FE SPACES ---
    ip_u = Lagrange{RefTriangle, 1}()^2
    ip_d = Lagrange{RefTriangle, 1}()
    qr   = QuadratureRule{RefTriangle}(2)

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
    u_max = 6.0e-3
    ch = ConstraintHandler(dh_u)
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> [0.0, 0.0]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "top"),    (x, t) -> [0.0, t]))
    close!(ch)

    # --- INITIAL CRACK (notch) ---
    addnodeset!(grid, "notch", x -> abs(x[2] - 0.5) < 1e-8 && x[1] <= 0.5 + 1e-8)
    ch_d = ConstraintHandler(dh_d)
    add!(ch_d, Dirichlet(:d, getnodeset(grid, "notch"), (x, t) -> 1.0))
    close!(ch_d)
    update!(ch_d, 0.0)

    # --- MATERIALS ---
    mat_elastic  = plane_strain_tensors(210.0e3, 0.3)
    mat_fracture = PhaseFieldMaterial(2.7e-3, 0.0150, 1e-7)

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

    # Apply initial crack
    apply!(d, ch_d)

    # --- STAGGERED LOOP ---
    n_steps = 200
    for step in 1:n_steps
        t = step * (u_max / n_steps)
        println("Step $step / $n_steps,  t = $t")
        update!(ch, t)

        update_history!(H, cellvalues_u, dh_u, u, mat_elastic)

        solve_phasefield!(K_d, f_d, cellvalues_d, dh_d, d, H, mat_fracture)
        apply!(d, ch_d)   # re-enforce notch every step

        solve_mechanics!(K_u, f_u, u, cellvalues_u, cellvalues_d, dh_u, dh_d, d, mat_elastic, mat_fracture, ch)
    end

    # --- OUTPUT ---
    VTKGridFile("phasefield_SENS", grid) do vtk
        write_solution(vtk, dh_u, u)
        write_solution(vtk, dh_d, d)
    end

    return u, d, dh_u, dh_d, grid
end



@profview for i in 1:3 run_phasefield(50, 50) end