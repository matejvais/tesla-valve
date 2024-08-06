from mesh import *

# obtain the mesh of an Tesla valve
msh = netgen_mesh(lobes=4, max_elem_size=10)


# define function spaces
V = VectorFunctionSpace(msh, "CG", 2)
W = FunctionSpace(msh, "CG", 1)
Z = V * W

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)



# Re = Constant(100.0)

# F = (
#     1.0 / Re * inner(grad(u), grad(v)) * dx +
#     inner(dot(grad(u), u), v) * dx -
#     p * div(v) * dx +
#     div(u) * q * dx
# )

# bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
#        DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

# nullspace = MixedVectorSpaceBasis(
#     Z, [Z.sub(0), VectorSpaceBasis(constant=True)])