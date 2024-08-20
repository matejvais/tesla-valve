from firedrake import *
from alfi import *
from defcon import *

import shutil

def reshape(s):
    if len(s) == 2:
        return as_tensor([[s[0], s[1]],
                          [s[1], -s[0]]])
    else:
        return as_tensor([[s[0], s[1], s[2]],
                          [s[1], s[3], s[4]],
                          [s[2], s[4], -s[0] - s[3]]])
def flatten(S):
    if S.ufl_shape == (2, 2):
        return as_vector([S[0, 0], S[0, 1]])
    else:
        return as_vector([S[0, 0], S[0, 1], S[0, 2], S[1, 1], S[1, 2]])


class NonNewtonianProblem(BifurcationProblem):
    def mesh(self, comm):
        # Markers: 10 = inlet, 11 = outlet, 12 = wall
        base = Mesh('mesh/pipe.msh', comm=comm)
        mh = BaryMeshHierarchy(base, 0)
        self.dim = base.geometric_dimension()
        return mh[-1]

    def function_space(self, mesh):
        dim = self.dim
        k = self.k = 2  # polynomial degree
        V = VectorFunctionSpace(mesh, "CG", k)
        Q = FunctionSpace(mesh, "DG", k-1)
        S = VectorFunctionSpace(mesh, "DG", k-1, dim=int(dim*(dim+1)/2 - 1))
        Z = MixedFunctionSpace([V, Q, S])
        return Z

    def parameters(self):
        mu = Constant(0)
        p = Constant(0)
        return [(p, "p", r"$p$"),
                (mu, "mu", r"$\mu$")]

    def residual(self, z, params, w):
        (u, p, s) = split(z)
        (v, q, t) = split(w)
        S = reshape(s)
        T = reshape(t)
        mesh = z.function_space().mesh()
        n = FacetNormal(mesh)

        alpha_star = Constant(0)
        p_ = params[0]
        mu = params[1]
        D = sym(grad(u))
        G = S - 2 * mu * (alpha_star + inner(D, D))**((p_-2)/2) * D

        F = (
              inner(S, sym(grad(v)))*dx
            - inner(outer(u, u), sym(grad(v)))*dx
            + inner(v, dot(outer(u, u), n))*ds
            - inner(p, div(v))*dx
            - inner(div(u), q)*dx
            + inner(G, T)*dx
            )

        return F

    def boundary_conditions(self, Z, params):
        mesh = Z.mesh()
        x = SpatialCoordinate(mesh)

        # Inlet BC
        bc_inflow = DirichletBC(Z.sub(0), as_vector([-(x[1] + 1) * (x[1] - 1), 0]), 10)

        # Wall
        bc_wall = DirichletBC(Z.sub(0), Constant((0, 0)), 12)

        bcs = [bc_inflow, bc_wall]

        return bcs

    def initial_guess(self, Z, params, n):
        # Solve Stokes in velocity-pressure form to make the initial guess
        W = MixedFunctionSpace([Z.sub(0), Z.sub(1)])
        z = Function(W)
        w = TestFunction(W)

        mu = params[1]
        Re = 1/mu

        (u, p) = split(z)
        (v, q) = split(w)
        F = (
              2.0/Re * inner(sym(grad(u)), sym(grad(v)))*dx
            - div(v)*p*dx
            - q*div(u)*dx
            )
        bcs = self.boundary_conditions(W, params)
        solve(F == 0, z, bcs, solver_parameters={"snes_monitor": None})

        z_ = Function(Z)
        z_.subfunctions[0].assign(z.subfunctions[0])
        z_.subfunctions[1].assign(z.subfunctions[1])
        S = 2*mu*sym(grad(z.subfunctions[0]))
        z_.subfunctions[2].interpolate(flatten(S))

        return z_

    def functionals(self):
        def sqL2(z, params):
            (u, p, S) = split(z)
            j = assemble(inner(u, u)*dx)
            return j

        return [(sqL2, "sqL2", r"$\|u\|^2$")]

    def solver_parameters(self, params, task, **kwargs):
        sp = {"snes_type": "newtonls",
              "snes_monitor": None,
              "snes_linesearch_type": "l2",
              "snes_linesearch_monitor": None,
              "snes_linesearch_maxstep": 1.0,
              "snes_converged_reason": None,
              "ksp_type": "preonly",
              "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps",
              "mat_mumps_icntl_14": 500}
        return sp

    def save_pvd(self, z, pvd, params):
        (u, p, s) = z.subfunctions
        mesh = z.function_space().mesh()
        S = project(reshape(s), TensorFunctionSpace(mesh, "DG", self.k-1))
        u.rename("Velocity"); p.rename("Pressure"); S.rename("Stress")
        pvd.write(u, p, S)

    def number_solutions(self, params):
        (p, mu) = params

        if p == 2:
            Re = 1/mu
            if   Re < 18:  return 1
            elif Re < 41:  return 3
            elif Re < 75:  return 5
            elif Re < 100: return 8
            else:          return float("inf")
        else:
            return float("inf")

    #def predict(self, *args, **kwargs):
    #    return secant(*args, **kwargs)


if __name__ == "__main__":
    dc = DeflatedContinuation(problem=NonNewtonianProblem(), teamsize=4, verbose=True, clear_output=True, logfiles=False)

    stages = [1, 2]  # 1, 2, 3, ...

    if 1 in stages:
        mus = list(1/Re for Re in linspace(1, 100, 2))
        dc.run(values={"p": 1.8, "mu": mus}, freeparam="mu")
        if COMM_WORLD.rank == 0:
            shutil.rmtree("output-stage-1")
            shutil.copytree("output", "output-stage-1")
        COMM_WORLD.barrier()

    if 2 in stages:
        dc.run(values={"p": linspace(1.8, 1.0, 3), "mu": mus[-1]}, freeparam="p")
        if COMM_WORLD.rank == 0:
            shutil.rmtree("output-stage-2")
            shutil.copytree("output", "output-stage-2")
        COMM_WORLD.barrier()

