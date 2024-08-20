from firedrake import *
from alfi import *

base = RectangleMesh(25, 5, 10, 1, diagonal="crossed")
mh = BaryMeshHierarchy(base, 0)
dim = base.geometric_dimension()
mesh = mh[-1]

k = 3 # polynomial degree
V = VectorFunctionSpace(mesh, "CG", k)
Q = FunctionSpace(mesh, "DG", k-1)
S = VectorFunctionSpace(mesh, "DG", k-1, dim=int(dim*(dim+1)/2 - 1))
Z = MixedFunctionSpace([V, Q, S])

def reshape(s):
    if dim == 2:
        return as_tensor([[s[0], s[1]],
                          [s[1], -s[0]]])
    else:
        return as_tensor([[s[0], s[1], s[2]],
                          [s[1], s[3], s[4]],
                          [s[2], s[4], -s[0] - s[3]]])

z = Function(Z)
(u, p, s) = split(z)
(v, q, t) = split(TestFunction(Z))
S = reshape(s)
T = reshape(t)
D = sym(grad(u))

alpha_star = Constant(0)
mu_star = Constant(1)
p_ = Constant(2)
G = S - 2 * mu_star * (alpha_star + inner(D, D))**((p_-2)/2) * D

F = (
      inner(S, sym(grad(v)))*dx
    - inner(outer(u, u), sym(grad(v)))*dx
    - inner(p, div(v))*dx
    - inner(div(u), q)*dx
    + inner(G, T)*dx
    )

(x, y) = SpatialCoordinate(mesh)
bcs = [DirichletBC(Z.sub(0), Constant((0, 0)), (3, 4)),
       DirichletBC(Z.sub(0), as_vector([4*y*(1-y), 0]), (1,)),
       DirichletBC(Z.sub(0).sub(1), Constant(0), (2,))]
z.subfunctions[0].interpolate(as_vector([4*y*(1-y), 0]))

sp = {"snes_type": "newtonls",
      "snes_monitor": None,
      "ksp_type": "preonly",
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps",
      "mat_mumps_icntl_14": 2000}

solve(F == 0, z, bcs, solver_parameters=sp)
(u, p, s) = z.subfunctions
S = project(reshape(s), TensorFunctionSpace(mesh, "DG", k-1))
u.rename("Velocity"); p.rename("Pressure"); S.rename("Stress")
File("output.pvd").write(u, p, S)
