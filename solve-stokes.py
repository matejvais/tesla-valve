from firedrake import *
import netgen_mesh as nm    # custom module for mesh generation

def solve_stokes(ngmsh, orientation=1, inlet_velocity_coef=0.1, name="output"):
    '''
    Copied (with minor modifications) from josef/sudden-expansion/sudden-expansion.py.
    Solve Stokes in velocity-pressure form to make the initial guess.
    '''

    mesh = Mesh(ngmsh)

    # function spaces
    dim = 2 # dimension of the domain
    k = 3 # polynomial degree
    V = VectorFunctionSpace(mesh, "CG", k)
    Q = FunctionSpace(mesh, "DG", k-1)
    Z = MixedFunctionSpace([V, Q])
    z = Function(Z)
    w = TestFunction(Z)
    
    # define boundary marks for the inlet, the outlet, and the wall
    labels_left = [i+1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name == "left"]
    labels_right = [i+1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name == "right"]
    labels_wall = [i+1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ["line","curve"]]

    # define boundary conditions
    (x, y) = SpatialCoordinate(mesh)
    if orientation == 1:    # fluid flows from left to right (->)
        print("Flow direction: =>")
        bc_in = DirichletBC(Z.sub(0), as_vector([-inlet_velocity_coef*y*(y+38),0]), labels_left)   # inflow velocity profile
        bc_out = DirichletBC(Z.sub(0).sub(1), Constant(0), labels_right)  # outflow
        z.subfunctions[0].interpolate(as_vector([-inlet_velocity_coef*y*(y+38), 0]))
    else: # fluid flows from right to left (<-)
        print("Flow direction: <=")
        n_curves = len(ngmsh.GetRegionNames(codim=1))   # number of curves forming the boundary of the valve
        if n_curves%2 == 1:    # even number of lobes
            bc_in = DirichletBC(Z.sub(0), as_vector([inlet_velocity_coef*(y+4)*(y+42),0]), labels_right)   # inflow velocity profile
            z.subfunctions[0].interpolate(as_vector([inlet_velocity_coef*(y+4)*(y+42),0]))
        else:   # odd number of lobes
            bc_in = DirichletBC(Z.sub(0), as_vector([inlet_velocity_coef*(y-4)*(y+34),0]), labels_right)
            z.subfunctions[0].interpolate(as_vector([inlet_velocity_coef*(y-4)*(y+34),0]))
        bc_out = DirichletBC(Z.sub(0).sub(1), Constant(0), labels_left)  # outflow
    bc_wall = DirichletBC(Z.sub(0), 0, labels_wall)    # zero velocity on the walls of the valve
    bcs = [bc_in, bc_wall, bc_out]

    Re = 1/100 # Reynolds number

    (u, p) = split(z)
    (v, q) = split(w)
    F = (
        2.0/Re * inner(sym(grad(u)), sym(grad(v)))*dx
        - div(v)*p*dx
        - q*div(u)*dx
        )
    solve(F == 0, z, bcs, solver_parameters={"snes_monitor": None})

    (u, p) = z.subfunctions
    u.rename("Velocity"); p.rename("Pressure")
    VTKFile(f"results-stokes/{name}.pvd").write(u, p)


if __name__ == "__main__":
    ngmsh = nm.netgen_mesh(lobes=4, max_elem_size=5)
    solve_stokes(ngmsh, orientation=1, inlet_velocity_coef=0.05)