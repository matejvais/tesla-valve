import firedrake as fd
import netgen_mesh as nm # custom module form netgen_mesh.py


def initial_guess(Z, bcs, nu):
    '''
    Copied (with minor modifications) from josef/sudden-expansion/sudden-expansion.py.
    '''
    # Solve Stokes in velocity-pressure form to make the initial guess
    W = fd.MixedFunctionSpace([Z.sub(0), Z.sub(1)])
    z = fd.Function(W)
    w = fd.TestFunction(W)

    Re = 1/nu

    (u, p) = fd.split(z)
    (v, q) = fd.split(w)
    F = (
            2.0/Re * fd.inner(fd.sym(fd.grad(u)), fd.sym(fd.grad(v)))*fd.dx
        - fd.div(v)*p*fd.dx
        - q*fd.div(u)*fd.dx
        )
    # bcs = self.boundary_conditions(W, params)
    fd.solve(F == 0, z, bcs, solver_parameters={"snes_monitor": None})

    # def flatten(S):
    #     return fd.as_vector([S[0, 0], S[0, 1]])

    z_ = fd.Function(Z)
    z_.subfunctions[0].assign(z.subfunctions[0])
    z_.subfunctions[1].assign(z.subfunctions[1])
    # S = 2*nu*fd.sym(fd.grad(z.subfunctions[0]))
    # z_.subfunctions[2].interpolate(flatten(S))

    return z_
    # return z


def solve_navier_stokes(ngmsh):
    '''
    From nmmo403/lecture5/ns_cylinder.py. Translated from Fenics to Firedrake by Chat GPT.
    '''
    mesh = fd.Mesh(ngmsh)

    # Define finite elements
    Ep = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
    Ev = fd.VectorElement("CG", mesh.ufl_cell(), 2)
    Evp = fd.MixedElement([Ev, Ep])

    # Build function spaces (Taylor-Hood)
    V = fd.FunctionSpace(mesh, Ev)
    P = fd.FunctionSpace(mesh, Ep)
    W = fd.FunctionSpace(mesh, Evp)

    # # No-slip boundary condition for velocity on walls and cylinder - boundary id 3
    # noslip = fd.Constant((0, 0))
    # bcv_walls = fd.DirichletBC(W.sub(0), noslip, bndry, 3)
    # bcv_cylinder = fd.DirichletBC(W.sub(0), noslip, bndry, 5)

    # define boundary conditions
    x = fd.SpatialCoordinate(mesh)
    labels_wall = [i+1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ["line","curve"]]
    labels_left = [i+1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name == "left"]
    labels_right = [i+1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name == "right"]
    bc_wall = fd.DirichletBC(W.sub(0), 0, labels_wall)    # zero velocity on the walls of the valve
    bc_in = fd.DirichletBC(W.sub(0), fd.as_vector([-0.1*x[1]*(x[1]+38),0]), labels_left)   # inflow velocity profile
    bc_out = fd.DirichletBC(W.sub(0).sub(1), fd.Constant(0), labels_right)
    bcs = [bc_wall, bc_in, bc_out]
    z = fd.Function(W)
    z.subfunctions[0].interpolate(fd.as_vector([-0.1*x[1]*(x[1]+38), 0]))

    # U = 1.5
    nu = fd.Constant(100)
    dt = 0.05
    t_end = 2
    theta = fd.Constant(0.5)   # Crank-Nicholson timestepping

    # # Inflow boundary condition for velocity - boundary id 1
    # v_in = fd.Expression(("U * 4.0 * x[1] * (0.41 - x[1]) / ( 0.41 * 0.41 )", "0.0"), U=U, degree=2)
    # bcv_in = fd.DirichletBC(W.sub(0), v_in, bndry, 1)

    # # Collect boundary conditions
    # bcs = [bcv_cylinder, bcv_walls, bcv_in]

    # Facet normal, identity tensor and boundary measure
    n = fd.FacetNormal(mesh)
    I = fd.Identity(mesh.geometric_dimension())
    # ds = fd.Measure("ds", subdomain_data=bndry)

    # Define unknown and test function(s)
    v_, p_ = fd.TestFunctions(W)

    # current unknown time step
    w = fd.Function(W)
    v, p = fd.split(w)

    # previous known time step
    w0 = fd.Function(W)
    v0, p0 = fd.split(w0)

    def a(v, u):
        D = fd.sym(fd.grad(v))
        return (fd.inner(fd.grad(v)*v, u) + fd.inner(2*nu*D, fd.grad(u)))*fd.dx

    def b(q, v):
        return fd.inner(fd.div(v), q)*fd.dx

    # variational form without time derivative in current time
    F1 = a(v, v_) - b(p_, v) - b(p, v_)

    # variational forms without time derivative in previous time
    F0 = a(v0, v_) - b(p_, v) - b(p, v_)

    # combine variational forms with time derivative
    #
    #  dw/dt + F(w,t) = 0 is approximated as
    #  (w-w0)/dt + theta*F(w,t) + (1-theta)*F(w0,t0) = 0
    #

    F = fd.Constant(1.0/dt)*fd.inner((v-v0), v_)*fd.dx + theta*F1 + (1.0-theta)*F0

    J = fd.derivative(F, w)

    problem = fd.NonlinearVariationalProblem(F, w, bcs=bcs, J=J)
    solver = fd.NonlinearVariationalSolver(problem)

    prm = solver.snes.ksp  # Solver parameters
    # Customize solver parameters (e.g., Newton's method)
    prm.rtol = 1e-12
    prm.atol = 1e-12
    prm.max_it = 20

    # Create files for storing solution
    name = "ns"
    out_file = dict()
    for i in ['v', 'p']:
        out_file[i] = fd.VTKFile(f"results_{name}/{i}.pvd")
        # out_file[i].parameters["flush_output"] = True

    v, p = w.split()
    v.rename("v", "velocity")
    p.rename("p", "pressure")

    # Time-stepping
    t = 0.0

    # Calculate the initial guess
    init = initial_guess(W, bcs, nu=nu)
    v_init, p_init = init.subfunctions[0], init.subfunctions[1]

    # Ensure consistency by interpolating the initial guess into the same space
    w_init = fd.Function(W)
    w_init.subfunctions[0].assign(v_init)
    w_init.subfunctions[1].assign(p_init)

    # Write initial conditions
    v, p = w_init.split()
    v.rename("v", "velocity")
    p.rename("p", "pressure")
    out_file['v'].write(v)
    out_file['p'].write(p)

    while t < t_end:

        fd.PETSc.Sys.Print("t =", t)

        # move current solution to previous slot w0
        w0.assign(w)

        # update time-dependent parameters
        t += dt

        # Compute
        # fd.begin("Solving ....")
        solver.solve()
        # fd.end()

        # Extract solutions:
        v, p = w.split()

        # # Report drag and lift
        # D = fd.sym(fd.grad(v))
        # T = -p*I + 2*nu*D
        # force = fd.dot(T, n)
        # D = -(2.0*force[0]/(1.0*1.0*0.1))*ds(5)
        # L = -(2.0*force[1]/(1.0*1.0*0.1))*ds(5)
        # drag.append((t, fd.assemble(D)))
        # lift.append((t, fd.assemble(L)))

        # Save to file
        out_file['v'].write(v)
        out_file['p'].write(p)

    # if fd.COMM_WORLD.rank == 0:
    #     import matplotlib
    #     matplotlib.use('Agg')
    #     import matplotlib.pyplot as plt

    #     drag = np.array(drag)
    #     lift = np.array(lift)
    #     plt.plot(drag[:, 0], drag[:, 1], '-', label='drag')
    #     plt.plot(lift[:, 0], lift[:, 1], '-', label='lift')
    #     plt.title('Flow around cylinder benchmark')
    #     plt.xlabel('time')
    #     plt.ylabel('lift/drag coeff')
    #     plt.legend(loc=1)
    #     plt.savefig('graph_{}.pdf'.format("lift_drag"), bbox_inches='tight')


if __name__ == "__main__":
    ngmsh = nm.netgen_mesh(lobes=4, max_elem_size=5)
    solve_navier_stokes(ngmsh)