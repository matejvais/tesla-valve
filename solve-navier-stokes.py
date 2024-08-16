import numpy as np
import firedrake as fd
import netgen_mesh


def solve_navier_stokes(ngmsh):
    mesh = Mesh(ngmsh)

    # Mesh generation using a custom module or directly using Firedrake
    import mymesh
    # (mesh, bndry) = mymesh.read_from_hdf5("bench_csg")
    m = mymesh.generate(n=3)
    (mesh, bndry) = m[-1]

    # Define finite elements
    Ep = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
    Ev = fd.VectorElement("CG", mesh.ufl_cell(), 2)
    Evp = fd.MixedElement([Ev, Ep])

    # Build function spaces (Taylor-Hood)
    V = fd.FunctionSpace(mesh, Ev)
    P = fd.FunctionSpace(mesh, Ep)
    W = fd.FunctionSpace(mesh, Evp)

    # No-slip boundary condition for velocity on walls and cylinder - boundary id 3
    noslip = fd.Constant((0, 0))
    bcv_walls = fd.DirichletBC(W.sub(0), noslip, bndry, 3)
    bcv_cylinder = fd.DirichletBC(W.sub(0), noslip, bndry, 5)

    U = 1.5
    nu = fd.Constant(0.001)
    dt = 0.1
    t_end = 15
    theta = fd.Constant(0.5)   # Crank-Nicholson timestepping

    # Inflow boundary condition for velocity - boundary id 1
    v_in = fd.Expression(("U * 4.0 * x[1] * (0.41 - x[1]) / ( 0.41 * 0.41 )", "0.0"), U=U, degree=2)
    bcv_in = fd.DirichletBC(W.sub(0), v_in, bndry, 1)

    # Collect boundary conditions
    bcs = [bcv_cylinder, bcv_walls, bcv_in]

    # Facet normal, identity tensor and boundary measure
    n = fd.FacetNormal(mesh)
    I = fd.Identity(mesh.geometric_dimension())
    ds = fd.Measure("ds", subdomain_data=bndry)

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
        out_file[i] = fd.XDMFFile(f"results_{name}/{i}.xdmf")
        out_file[i].parameters["flush_output"] = True

    v, p = w.split(True)
    v.rename("v", "velocity")
    p.rename("p", "pressure")

    # Time-stepping
    t = 0.0

    # Save initial conditions
    out_file['v'].write(v, t)
    out_file['p'].write(p, t)

    lift = []
    drag = []

    while t < t_end:

        fd.comm.world.rank_zero_print("t =", t)

        # move current solution to previous slot w0
        w0.assign(w)

        # update time-dependent parameters
        t += dt

        # Compute
        fd.begin("Solving ....")
        solver.solve()
        fd.end()

        # Extract solutions:
        v, p = w.split(True)

        # Report drag and lift
        D = fd.sym(fd.grad(v))
        T = -p*I + 2*nu*D
        force = fd.dot(T, n)
        D = -(2.0*force[0]/(1.0*1.0*0.1))*ds(5)
        L = -(2.0*force[1]/(1.0*1.0*0.1))*ds(5)
        drag.append((t, fd.assemble(D)))
        lift.append((t, fd.assemble(L)))

        # Save to file
        out_file['v'].write(v, t)
        out_file['p'].write(p, t)

    if fd.COMM_WORLD.rank == 0:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        drag = np.array(drag)
        lift = np.array(lift)
        plt.plot(drag[:, 0], drag[:, 1], '-', label='drag')
        plt.plot(lift[:, 0], lift[:, 1], '-', label='lift')
        plt.title('Flow around cylinder benchmark')
        plt.xlabel('time')
        plt.ylabel('lift/drag coeff')
        plt.legend(loc=1)
        plt.savefig('graph_{}.pdf'.format("lift_drag"), bbox_inches='tight')
