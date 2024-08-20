import numpy as np
from dolfin import *
import mshr

comm = MPI.comm_world
rank = MPI.rank(comm)

parameters["std_out_all_processes"] = False

import mymesh
# (mesh, bndry) = mymesh.read_from_hdf5("bench_csg")
m=mymesh.generate(n=3)
(mesh, bndry) = m[-1]

# Define finite elements
Ep = FiniteElement("CG",mesh.ufl_cell(),1)
Ev = VectorElement("CG",mesh.ufl_cell(),2)
Evp=MixedElement([Ev,Ep]) 

# Build function spaces (Taylor-Hood)
V = FunctionSpace(mesh,Ev)
P = FunctionSpace(mesh,Ep)
W = FunctionSpace(mesh,Evp)

# No-slip boundary condition for velocity on walls and cylinder - boundary id 3
noslip = Constant((0, 0))
bcv_walls = DirichletBC(W.sub(0), noslip, bndry, 3)
bcv_cylinder = DirichletBC(W.sub(0), noslip, bndry, 5)

U=1.5
nu = Constant(0.001)
dt = 0.1
t_end = 15
theta=Constant(0.5)   # Crank-Nicholson timestepping

# Inflow boundary condition for velocity - boundary id 1
v_in = Expression(("U * 4.0 * x[1] * (0.41 - x[1]) / ( 0.41 * 0.41 )", "0.0"), U=U, degree=2)
bcv_in = DirichletBC(W.sub(0), v_in, bndry, 1)

# Collect boundary conditions
bcs = [bcv_cylinder, bcv_walls, bcv_in]

# Facet normal, identity tensor and boundary measure
n = FacetNormal(mesh)
I = Identity(mesh.geometry().dim())
ds = Measure("ds", subdomain_data=bndry)

# Define unknown and test function(s)
(v_, p_) = TestFunctions(W)

# current unknown time step
w = Function(W)
(v, p) = split(w)

# previous known time step
w0 = Function(W)
(v0, p0) = split(w0)

def a(v,u) :
    D = sym(grad(v))
    return (inner(grad(v)*v, u) + inner(2*nu*D, grad(u)))*dx

def b(q,v) :
    return inner(div(v),q)*dx

# variational form without time derivative in current time
F1 = a(v,v_) - b(p_,v) - b(p,v_) 

# variational forms without time derivative in previous time
F0 = a(v0,v_) - b(p_,v) - b(p,v_)

#combine variational forms with time derivative
#
#  dw/dt + F(w,t) = 0 is approximated as
#  (w-w0)/dt + theta*F(w,t) + (1-theta)*F(w0,t0) = 0
#

F = Constant(1.0/dt)*inner((v-v0),v_)*dx + theta*F1 + (1.0-theta)*F0 

J = derivative(F, w)

problem=NonlinearVariationalProblem(F,w,bcs,J)
solver=NonlinearVariationalSolver(problem)

prm = solver.parameters
#info(prm,True)  #get full info on the parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['absolute_tolerance'] = 1E-12
prm['newton_solver']['relative_tolerance'] = 1e-12
prm['newton_solver']['maximum_iterations'] = 20
prm['newton_solver']['linear_solver'] = 'mumps'


# Create files for storing solution
name="ns"
out_file=dict()
for i in ['v', 'p'] :
    out_file[i] = XDMFFile(f"results_{name}/{i}.xdmf")
    out_file[i].parameters["flush_output"] = True

(v,p)=w.split(True)
v.rename("v", "velocity")
p.rename("p", "pressure")

# Time-stepping
t = 0.0

# Extract solutions:
assign(v, w.sub(0))
assign(p, w.sub(1))

# Save to file
out_file['v'].write(v, t)
out_file['p'].write(p, t)

lift=[]
drag=[]

while t < t_end:

    if rank==0: print("t =", t)

    # move current solution to previous slot w0
    w0.assign(w)

    # update time-dependent parameters
    t += dt

    # Compute
    begin("Solving ....")
    solver.solve()
    end()

    # Extract solutions:
    assign(v, w.sub(0))
    assign(p, w.sub(1))

    # Report drag and lift
    D=sym(grad(v))
    T= -p*I + 2*nu*D
    force = dot(T, n)
    D = -(2.0*force[0]/(1.0*1.0*0.1))*ds(5)
    L = -(2.0*force[1]/(1.0*1.0*0.1))*ds(5)
    drag.append( (t, assemble(D)) )
    lift.append( (t,assemble(L)) )
                            
    
    # Save to file
    out_file['v'].write(v, t)
    out_file['p'].write(p, t)

if rank == 0 :
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    drag=np.array(drag)
    lift=np.array(lift)
    plt.plot(drag[:,0], drag[:,1], '-', label='drag')
    plt.plot(lift[:,0], lift[:,1], '-', label='lift')
    plt.title('flow around cylinder benchmark')
    plt.xlabel('time')
    plt.ylabel('lift/drag coeff')
    plt.legend(loc=1)
    plt.savefig('graph_{}.pdf'.format("lift_drag"), bbox_inches='tight')
