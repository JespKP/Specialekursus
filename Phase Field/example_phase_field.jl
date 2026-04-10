function run_phasefield(nx::Int, ny::Int)

    # --- MESH ---
    grid = generate_grid(Quadrilateral, (nx, ny), Vec(0.0, 0.0), Vec(1.0, 1.0))

    # --- FE SPACES ---
    ip_u = Lagrange{RefQuadrilateral, 1}()^2
    ip_d = Lagrange{RefQuadrilateral, 1}()
    qr   = QuadratureRule{RefQuadrilateral}(2)

    cellvalues_u = CellValues(qr, ip_u)
    cellvalues_d = CellValues(qr, ip_d)

    # --- DOF HANDLERS ---
    dh_u = DofHandler(grid)
    add!(dh_u, :u, ip_u)
    close!(dh_u)

    dh_d = DofHandler(grid)
    add!(dh_d, :d, ip_d)
    close!(dh_d)

    # --- BOUNDARY CONDITIONS ON u ---
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
    mat_fracture = PhaseFieldMaterial(2.7e-3, 0.0075, 1e-7)

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
    n_steps = 1000
    for step in 1:n_steps
        t = step * (u_max / n_steps)
        update!(ch, t)

        update_history!(H, cellvalues_u, dh_u, u, mat_elastic)
        d .= solve_phasefield!(K_d, f_d, cellvalues_d, dh_d, H, mat_fracture)
        apply!(d, ch_d)   # re-enforce notch — d must stay 1 on crack
        u .= solve_mechanics!(K_u, f_u, cellvalues_u, cellvalues_d, dh_u, dh_d, d, mat_elastic, mat_fracture, ch)
    end

    # --- OUTPUT ---
    VTKGridFile("phasefield_SENS", grid) do vtk
        write_solution(vtk, dh_u, u)
        write_solution(vtk, dh_d, d)
    end

    return u, d, dh_u, dh_d, grid;
end

Base.invokelatest(run_phasefield, 100, 100);