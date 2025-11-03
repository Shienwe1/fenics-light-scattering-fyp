from mpi4py import MPI
from dolfinx import mesh
import numpy

# Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

# Define function space
from dolfinx import fem
V = fem.functionspace(domain, ("Lagrange", 1))

# Define Dirichlet boundary condition
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

# define boundary condition
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

# Locate boundary dofs
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc= fem.dirichletbc(uD, boundary_dofs)

# Define trial and test functions
import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define source term
from dolfinx.fem import default_scalar_type
f=fem.Constant(domain, default_scalar_type( -6.0))

# Define variational problem
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Express inner product
from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Compute error
V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

error_max = numpy.max(numpy.abs(uh.x.array - uex.x.array))
if domain.comm.rank == 0:
    print(f"L2 error: {error_L2}")
    print(f"Max error: {error_max}")

import pyvista
print(pyvista.global_theme.jupyter_backend)

from dolfinx import plot
domain.topology.create_connectivity(0, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, scalars=uh.x.array, cmap="viridis")
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("solution.png")

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()

warped = u_grid.warp_by_vector("u", factor=0.1)
plotter2 = pyvista.Plotter()
plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
if not pyvista.OFF_SCREEN:
    plotter2.show()
