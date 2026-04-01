using Ferrite, SparseArrays, SpecialeKursus

function run_example(nx::Int, ny::Int)
    # Geometry and mesh
    L = 20.0
    grid = generate_grid(Quadrilateral, (nx, ny), Vec(0.0, 0.0), Vec(L, 1.0))
    addfacetset!(grid, "fixed", x -> abs(x[1]) < 1e-8)
    addfacetset!(grid, "rightt", x -> abs(x[1] - L) < 1e-8)

    # Finite element spaces
    dim = 2
    ip = Lagrange{RefQuadrilateral, 1}()^dim
    qr = QuadratureRule{RefQuadrilateral}(2)
    qr_face = FacetQuadratureRule{RefQuadrilateral}(1)

    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_face, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "fixed"), (x, t) -> [0.0, 0.0]))
    close!(ch)

    # Material model
    E = 200.0e3
    ν = 0.3
    mat = plane_stress_tensors(E, ν)

    # Plastic solver settings
    total_traction = x -> Vec(0.0, -1.0e2)
    n_increments = 100
    h = 100.0
    σ_y0 = 250.0

    # Integration scheme
    scheme = StandardIntegration(cellvalues)

    # Run simulation
    u, all_states = run_simulation_material(
        dh,
        scheme,
        mat,
        ch,
        getfacetset(grid, "rightt"),
        facetvalues,
        total_traction;
        model = :perfect_plasticity,
        n_increments = n_increments,
        h = h,
        σ_y0 = σ_y0
    )

    println("Mesh size: $(nx) x $(ny)  ($(getncells(grid)) cells, $(ndofs(dh)) dofs)")
    println("Max nodal displacement magnitude: ", maximum(abs.(u)))

    if all_states === nothing
        println("No plastic state stored for elastic material.")
    else
        plastic_steps = sum(state.Δλ > 0.0 for cell_states in all_states for state in cell_states)
        println("Number of Gauss-point plastic steps: ", plastic_steps)

        if !isempty(all_states)
            first_sigma = all_states[1][1].σ
            println("First Gauss-point stress tensor: ", first_sigma)
        end
    end

    VTKGridFile("plastic_solution", dh) do vtk
        write_solution(vtk, dh, u)
    end
    println("Wrote VTK output to plastic_solution.vtu")

    return u, all_states, dh, grid
end

@time run_example(20,20);

@profview for i in 1:3 run_example(20,20) end