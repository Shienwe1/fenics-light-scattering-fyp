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
