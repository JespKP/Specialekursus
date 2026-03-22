################################################################################
# FacetValues #
################################################################################
# Linear quad element with 2x2 integration points
ip = Lagrange{RefQuadrilateral, 1}()
qr = QuadratureRule{RefQuadrilateral}(2)
cv = CellValues(qr, ip, ip)

qr_face = FacetQuadratureRule{RefQuadrilateral}(2)
fv = FacetValues(qr_face, ip)

# distort the element a little (see notes)
xe = Ferrite.reference_coordinates(ip)
xe[4] = Tensors.Vec(0.0, 1.0)

for facet_id in 1:nfacets(ip)
    reinit!(fv, xe, facet_id)
    l_facet = sum(i->getdetJdV(fv, i), 1:getnquadpoints(fv))
    @show facet_id
    @show l_facet
end

qp=1
getnormal(fv, qp)


# vector valued interpolations

# Linear quad element with 2x2 integration points
ip = Lagrange{RefQuadrilateral, 1}()
qr = QuadratureRule{RefQuadrilateral}(2)
cv_scalar = CellValues(qr, ip, ip)

cv_scalar.fun_values.Nξ

cv_vector = CellValues(qr, ip^2, ip^2)
cv_vector.fun_values.Nξ
cv_vector.fun_values.Nξ

# geometry interpolations & element types in the grid

linear_grid = generate_grid(Quadrilateral, (1,1))
linear_grid.nodes

quadratic_grid = generate_grid(QuadraticQuadrilateral, (1,1))
quadratic_grid.nodes

refshape = RefQuadrilateral # only says it is a quadrilateral, no order attached
ip_function = Lagrange{refshape, 1}()
ip_geometry = Lagrange{refshape, 2}()
qr = QuadratureRule{refshape}(2)

cv = CellValues(qr, ip_function, ip_geometry)

xe_linear = getcoordinates(linear_grid, 1)
xe_quadratic = getcoordinates(quadratic_grid, 1)

reinit!(cv, xe_linear)
reinit!(cv, xe_quadratic)


ip_geometry = Serendipity{refshape, 2}()
cv_serendipity = CellValues(qr, ip_function, ip_geometry)
reinit!(cv, xe_quadratic)


# References and copies
grid = generate_grid(Quadrilateral, (1,1))

grid.nodes
grid.nodes[1] = Node(Vec(99.0, 99.0))

dh = DofHandler(grid)
add!(dh, :a, Lagrange{RefQuadrilateral,1}())
close!(dh)

dh.grid.nodes
grid.nodes[1] = Node(Vec(55.0, 99.0))
dh.grid.nodes

grid_copy = deepcopy(grid)
grid === grid_copy
grid_copy.nodes[1] = Node(Vec(22., 22.))