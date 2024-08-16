from firedrake import *
import netgen_mesh as nm    # custom module for mesh generation

def solve_stokes_mixed(ngmsh, power_law_index=2.0, poly_deg=3, name="output"):
    '''
    Solves the Stokes equation in the Tesla valve using the mixed formulation.
    Mostly used the code from josef/steady.py with some modifications:
        https://bitbucket.org/pefarrell/josef/src/master/
    Variables:
        ngmesh - Netgen mesh
        power_law_index - power in the constitutive relation (power-law fluid),
        poly_deg - polynomial degree of the finite element approximation.
        name - name of the saved solution file
    '''

    mesh = Mesh(ngmsh)

    # function spaces
    dim = 2 # dimension of the problem
    k = poly_deg # polynomial degree
    V = VectorFunctionSpace(mesh, "CG", k)
    Q = FunctionSpace(mesh, "DG", k-1)
    S = VectorFunctionSpace(mesh, "DG", k-1, dim=int(dim*(dim+1)/2 - 1))
    Z = MixedFunctionSpace([V, Q, S])

    def reshape(s):
        return as_tensor([[s[0], s[1]],
                        [s[1], -s[0]]])

    # variables
    z = Function(Z)
    (u, p, s) = split(z) # trial functions: velocity, pressure, stress
    (v, q, t) = split(TestFunction(Z)) # test functions: velocity, pressure, stress
    S = reshape(s)
    T = reshape(t)
    D = sym(grad(u))

    # the constitutive relation
    mu_zero = Constant(100) # dynamic viscosity
    p_ = Constant(power_law_index)
    G = S - 2 * mu_zero * (inner(D, D))**((p_-2)/2) * D

    # PDE residual
    F = (
        inner(S, sym(grad(v)))*dx
        - inner(outer(u, u), sym(grad(v)))*dx
        - inner(p, div(v))*dx
        - inner(div(u), q)*dx
        + inner(G, T)*dx
        )

    # define boundary conditions
    x = SpatialCoordinate(mesh)
    labels_wall = [i+1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ["line","curve"]]
    labels_in = [i+1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name == "inlet"]
    labels_out = [i+1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name == "outlet"]
    bc_wall = DirichletBC(Z.sub(0), 0, labels_wall)    # zero velocity on the walls of the valve
    bc_in = DirichletBC(Z.sub(0), as_vector([0.1*x[1]*(x[1]+38),0]), labels_in)   # inflow velocity profile
    bc_out = DirichletBC(Z.sub(0).sub(1), Constant(0), labels_out)
    bcs = [bc_wall, bc_in, bc_out]
    z.subfunctions[0].interpolate(as_vector([0.1*x[1]*(x[1]+38), 0]))

    # solver parameters
    sp = {"snes_type": "newtonls",
        "snes_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": 2000}

    # solve and save
    solve(F == 0, z, bcs, solver_parameters=sp)
    (u, p, s) = z.subfunctions
    S = project(reshape(s), TensorFunctionSpace(mesh, "DG", k-1))
    u.rename("Velocity"); p.rename("Pressure"); S.rename("Stress")
    VTKFile(f"{name}.pvd").write(u, p, S)


if __name__ == "__main__":
    ngmsh = nm.netgen_mesh(lobes=4, max_elem_size=10)
    solve_stokes_mixed(ngmsh)