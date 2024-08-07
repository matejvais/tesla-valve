# import PIL as pil
import numpy as np
#import nanomesh as nm
from netgen.geom2d import SplineGeometry
from firedrake import *




# BEGINNING OF NANOMESH FUNCTIONS
#################################################

# def generate_bw(file_name):
#     '''
#     Generate a black and white image and save it.
#     '''
#     image = pil.Image.open(file_name)
#     image = image.convert('L') 
#     image = image.point(lambda i: i < 175 and 255)
#     image.save("bw-valve-figure.png")

# def generate_array(file_name, save=False, display_np=False):
#     '''
#     Generate a numpy array from an image.
#     options:
#         save - saves the numpy array as a binary file
#         display_np - saves the visual representation of the numpy array as a image
#     '''
#     np_image = np.asarray(pil.Image.open(file_name))[:,:,0:3]
#     np_image = np.max(np_image, axis=2)
#     #print(np_image.shape)
#     if save:
#         np.save("np_image", np_image)
#     if display_np:
#         pil.Image.fromarray(np_image).save('visual_np_image.png')

# def nanomesh(file_name):
#     '''
#     Generation of the mesh from a numpy array.
#     See the following website for details:
#         https://nanomesh.readthedocs.io/en/latest/examples/examples_generate_a_2d_triangular_mesh.html
#     '''
#     data = nm.Image.load(file_name)
#     mesher = nm.Mesher2D(data)
#     mesher.generate_contour(max_edge_dist=5)
#     mesh = mesher.triangulate(opts='q20a10')
#     #triangle_mesh = mesh.get('triangle')
#     mesh.write('out.msh', file_format='gmsh22', binary=False)

# END OF NANOMESH FUNCTIONS
#################################################


def corner(pointA,pointB):
    '''
    Return the coordinates of the "corner" point used for creating a spline3 curve approximating a
    quarter-circle.
    '''
    a1,a2,b1,b2 = pointA[0],pointA[1],pointB[0],pointB[1]
    if b2 > a2:
        c1, c2 = 0.5 * (a1 + b1 + b2 - a2), 0.5 * (a2 + b2 + a1 - b1)
    else:
        c1, c2 = 0.5 * (a1 + b1 + a2 - b2), 0.5 * (a2 + b2 + b1 - a1)
    return tuple((c1, c2))


def generate_points(lobes):
    '''
    Returns an 2D array of points defining the Tesla valve. The first lobe is facing downward.
    Variables:
        lobes - number of lobes in the Tesla valve
    '''
    coords = np.zeros((lobes, 12, 2))
    x_shift, y_shift = 133, -38
    ps = np.array([
     [0   ,-38],
     [133 ,  0],
     [189 ,-16],
     [189 ,-24],
     [195 ,-26],
     [210 ,-53],
     [181 ,-69],
     [ 86 ,-37],
     [129 ,-26],
     [191 ,-37],
     [197 ,-49],
     [184 ,-57]])
    for l in range(lobes):
        qs = np.copy(ps)
        qs[:,1] *= (-1)**l  # mirror the y coordinates for odd lobes
        qs[:,0] += x_shift*l    # shift by the length of one lobe l times in the x direction
        qs[:,1] += y_shift*(l%2)    # shift odd lobes up in the y direction
        coords[l, :, :] = qs
    return coords


def netgen_mesh(lobes, max_elem_size, generate_pvd=True, name="tesla-valve.pvd"):
    '''
    Generates mesh in the shape of a Tesla valve with a given number of lobes.
    Variables:
        lobes - number of lobes in the valve (integer >= 2)
        max_elem_size - maximal size of a finite element forming the mesh
        name - name of the resulting mesh file
    Returns: mesh object
    '''
    geo = SplineGeometry()
    coords = generate_points(lobes)
    p = np.empty((lobes, 16), dtype=int)

    # add the beginning of the valve (inlet)
    bs = [(-45, 0), (-45, -38)] # define points
    b0, b1 = [geo.AppendPoint(*b) for b in bs]  # append points to the geometry
    p[0,0] = geo.AppendPoint(*coords[0,0])
    beginning_curves = [    # define curves that constitute the initial part of the domain boundary
        [["line", b0, b1], "inlet"],    # this part of the boundary forms an inlet for the fluid coming inside
        [["line", b1, p[0,0]], "line"]
    ]
    [geo.Append(c, bc=bc) for c, bc in beginning_curves]

    # add lobes
    for l in range(lobes):

        # defining points
        ps = [coords[l][0], coords[l][1], coords[l][2], coords[l][3], coords[l][4], coords[l][5], coords[l][6]] # outer boundary points
        qs = [coords[l][7], coords[l][8], coords[l][9], coords[l][10], coords[l][11]]   # inner boundary points
        
        # the first lobe
        if l == 0:  # for the first lobe, the point p[0,0] has already been appended above
            p[l,1], p[l,2], p[l,3], p[l,4], p[l,5], p[l,6] = [geo.AppendPoint(*p) for p in ps[1:7]]
            p[l,7], p[l,8], p[l,9], p[l,10], p[l,11] = [geo.AppendPoint(*q) for q in qs]
            p[l,12] = geo.AppendPoint(*corner(ps[4], ps[5]))    # p45
            p[l,13] = geo.AppendPoint(*corner(ps[5], ps[6]))    # p56
            p[l,14] = geo.AppendPoint(*corner(qs[2], qs[3]))    # q23
            p[l,15] = geo.AppendPoint(*corner(qs[3], qs[4]))    # q34

            outer_curves = [    # curves forming the outer boundary of a lobe
                [["line", p[l,0], p[l,6]], "line"],
                [["spline3", p[l,6], p[l,13], p[l,5]], "curve"],
                [["spline3", p[l,5], p[l,12], p[l,4]], "curve"],
                [["line", p[l,4], p[l,3]], "line"],
                [["line", p[l,3], p[l,2]], "line"],
                [["line", p[l,1], b0], "line"]
            ]
            inner_curves = [    # curves forming the inner boundary of a lobe
                [["line", p[l,7], p[l,8]], "line"],
                [["line", p[l,8], p[l,9]], "line"],
                [["spline3", p[l,9], p[l,14], p[l,10]], "curve"],
                [["spline3", p[l,10], p[l,15], p[l,11]], "curve"],
                [["line", p[l,11], p[l,7]], "line"] 
            ]
            
            # appending boundary curves to the geometry
            [geo.Append(c, bc=bc) for c, bc in outer_curves]
            [geo.Append(c, bc=bc) for c, bc in inner_curves]

        # middle lobes
        elif l <= lobes-2:  # lobes in the middle
            if l%2 == 0:    # lobes with an even index
                p[l,0], p[l,1], p[l,2], p[l,3], p[l,4], p[l,5], p[l,6] = [geo.AppendPoint(*p) for p in ps]
                p[l,7], p[l,8], p[l,9], p[l,10], p[l,11] = [geo.AppendPoint(*q) for q in qs]
                p[l,12] = geo.AppendPoint(*corner(ps[4], ps[5]))    # p45
                p[l,13] = geo.AppendPoint(*corner(ps[5], ps[6]))    # p56
                p[l,14] = geo.AppendPoint(*corner(qs[2], qs[3]))    # q23
                p[l,15] = geo.AppendPoint(*corner(qs[3], qs[4]))    # q34

                outer_curves = [
                    [["line", p[l-1, 1], p[l,6]], "line"],
                    [["spline3", p[l,6], p[l,13], p[l,5]], "curve"],
                    [["spline3", p[l,5], p[l,12], p[l,4]], "curve"],
                    [["line", p[l,4], p[l,3]], "line"],
                    [["line", p[l,3], p[l,2]], "line"],
                    [["line", p[l,1], p[l-1,2]], "line"]    # line connecting adjacent lobes
                ]
                inner_curves = [
                    [["line", p[l,7], p[l,8]], "line"],
                    [["line", p[l,8], p[l,9]], "line"],
                    [["spline3", p[l,9], p[l,14], p[l,10]], "curve"],
                    [["spline3", p[l,10], p[l,15], p[l,11]], "curve"],
                    [["line", p[l,11], p[l,7]], "line"] 
                ]

                # appending boundary curves to the geometry
                [geo.Append(c, bc=bc) for c, bc in outer_curves]
                [geo.Append(c, bc=bc) for c, bc in inner_curves]    

            else:   # lobes with odd indices have an opposite orientation of boundary curves
                p[l,1], p[l,2], p[l,3], p[l,4], p[l,5], p[l,6] = [geo.AppendPoint(*p) for p in ps[1:7]]
                p[l,7], p[l,8], p[l,9], p[l,10], p[l,11] = [geo.AppendPoint(*q) for q in qs]
                p[l,12] = geo.AppendPoint(*corner(ps[4], ps[5]))    # p45
                p[l,13] = geo.AppendPoint(*corner(ps[5], ps[6]))    # p56
                p[l,14] = geo.AppendPoint(*corner(qs[2], qs[3]))    # q23
                p[l,15] = geo.AppendPoint(*corner(qs[3], qs[4]))    # q34

                outer_curves = [
                    [["line", p[l,6], p[l-1,1]], "line"],
                    [["spline3", p[l,5], p[l,13], p[l,6]], "curve"],
                    [["spline3", p[l,4], p[l,12], p[l,5]], "curve"],
                    [["line", p[l,3], p[l,4]], "line"],
                    [["line", p[l,2], p[l,3]], "line"],
                    [["line", p[l-1,2], p[l,1]], "line"]    # line connecting adjacent lobes
                ]
                inner_curves = [
                    [["line", p[l,8], p[l,7]], "line"],
                    [["line", p[l,9], p[l,8]], "line"],
                    [["spline3", p[l,10], p[l,14], p[l,9]], "curve"],
                    [["spline3", p[l,11], p[l,15], p[l,10]], "curve"],
                    [["line", p[l,7], p[l,11]], "line"] 
                ]

                # appending boundary curves to the geometry
                [geo.Append(c, bc=bc) for c, bc in outer_curves]
                [geo.Append(c, bc=bc) for c, bc in inner_curves]

        # the final lobe
        else:
            if l%2 == 0:    # the final lobe has an even index
                p[l,1], p[l,5], p[l,6] = geo.AppendPoint(*ps[1]), geo.AppendPoint(*ps[5]), geo.AppendPoint(*ps[6])
                p[l,7], p[l,8], p[l,9], p[l,10], p[l,11] = [geo.AppendPoint(*q) for q in qs]
                p[l,13] = geo.AppendPoint(*corner(ps[5], ps[6]))    # p56
                p[l,14] = geo.AppendPoint(*corner(qs[2], qs[3]))    # q23
                p[l,15] = geo.AppendPoint(*corner(qs[3], qs[4]))    # q34
                outer_curves = [
                    [["line", p[l-1, 1], p[l,6]], "line"],
                    [["spline3", p[l,6], p[l,13], p[l,5]], "curve"],
                    [["line", p[l,1], p[l-1,2]], "line"]    # line connecting adjacent lobes
                ]
                inner_curves = [
                    [["line", p[l,7], p[l,8]], "line"],
                    [["line", p[l,8], p[l,9]], "line"],
                    [["spline3", p[l,9], p[l,14], p[l,10]], "curve"],
                    [["spline3", p[l,10], p[l,15], p[l,11]], "curve"],
                    [["line", p[l,11], p[l,7]], "line"] 
                ]
                # appending boundary curves to the geometry
                [geo.Append(c, bc=bc) for c, bc in outer_curves]
                [geo.Append(c, bc=bc) for c, bc in inner_curves]
                           
            else:   # the final lobe has an odd index, boundary curves are constructed with the opposite orientation  
                p[l,1], p[l,5], p[l,6] = geo.AppendPoint(*ps[1]), geo.AppendPoint(*ps[5]), geo.AppendPoint(*ps[6])
                p[l,7], p[l,8], p[l,9], p[l,10], p[l,11] = [geo.AppendPoint(*q) for q in qs]
                p[l,13] = geo.AppendPoint(*corner(ps[5], ps[6]))    # p56
                p[l,14] = geo.AppendPoint(*corner(qs[2], qs[3]))    # q23
                p[l,15] = geo.AppendPoint(*corner(qs[3], qs[4]))    # q34
                outer_curves = [
                    [["line", p[l,6], p[l-1, 1]], "line"],
                    [["spline3", p[l,5], p[l,13], p[l,6]], "curve"],
                    [["line", p[l-1,2], p[l,1]], "line"]    # line connecting adjacent lobes
                ]
                inner_curves = [
                    [["line", p[l,8], p[l,7]], "line"],
                    [["line", p[l,9], p[l,8]], "line"],
                    [["spline3", p[l,10], p[l,14], p[l,9]], "curve"],
                    [["spline3", p[l,11], p[l,15], p[l,10]], "curve"],
                    [["line", p[l,7], p[l,11]], "line"] 
                ]
                # appending boundary curves to the geometry
                [geo.Append(c, bc=bc) for c, bc in outer_curves]
                [geo.Append(c, bc=bc) for c, bc in inner_curves]

    # add the end of the valve (outlet)
    ref = coords[lobes-1][1]   # p1 point of the last lobe, used for reference coordinates
    if lobes % 2 == 1:
        es = [ref+[90,-39], ref+[94,-34], ref+[100,-34], ref+[148,-34], ref+[148,4], ref+[103,4]]
        e0, e1, e2, e3, e4, e5 = [geo.AppendPoint(*e) for e in es]
        end_curves = [
                [["line", p[lobes-1,5], e0], "line"],
                [["spline3", e0, e1, e2], "curve"],
                [["line", e2, e3], "line"],
                [["line", e3, e4], "outlet"],   # this part of the boundary forms an outlet for the fluid inside
                [["line", e4, e5], "line"],
                [["line", e5, p[lobes-1,1]], "line"]
        ]
    else:   # if the final lobe has an odd index (i.e. number of lobes is even), y coordinates must be mirrored
        es = [ref+[90,39], ref+[94,34], ref+[100,34], ref+[148,34], ref+[148,-4], ref+[103,-4]]
        e0, e1, e2, e3, e4, e5 = [geo.AppendPoint(*e) for e in es]
        end_curves = [
                [["line", e0, p[lobes-1,5]], "line"],
                [["spline3", e2, e1, e0], "curve"],
                [["line", e3, e2], "line"],
                [["line", e4, e3], "outlet"],   # this part of the boundary forms an outlet for the fluid inside
                [["line", e5, e4], "line"],
                [["line", p[lobes-1,1], e5], "line"]
        ]
    [geo.Append(c, bc=bc) for c, bc in end_curves]

    # construct the mesh
    ngmsh = geo.GenerateMesh(maxh=max_elem_size)

    # generate a .pvd file if required
    if generate_pvd: 
        msh = Mesh(ngmsh)
        VTKFile(f"output/{name}").write(msh)

    return ngmsh

def solve_ns(ngmsh):
    msh = Mesh(ngmsh)

    # define function spaces
    V = VectorFunctionSpace(msh, "CG", 2)
    W = FunctionSpace(msh, "CG", 1)
    Z = V * W

    # define boundary conditions
    x = SpatialCoordinate(msh)
    labels_wall = [i+1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ["line","curve"]]
    labels_in = [i+1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name == "inlet"]
    bc_wall = DirichletBC(V, 0, labels_wall)    # zero velocity on the walls of the valve
    bc_in = DirichletBC(V, 0.1*x[1]*(x[1]+38), labels_in)   # inflow velocity profile
    # Firedrake automatically prescribes the natural boundary condition (zero normal stress) on the outflow

    up = Function(Z)
    u, p = split(up)
    v, q = TestFunctions(Z)

    # define PDE residual
    Re = Constant(100.0)
    F = (
        1.0 / Re * inner(grad(u), grad(v)) * dx +
        inner(dot(grad(u), u), v) * dx -
        p * div(v) * dx +
        div(u) * q * dx
    )


if __name__ == "__main__":
    ngmsh = netgen_mesh(lobes=2, max_elem_size=10)
    solve_ns(ngmsh)