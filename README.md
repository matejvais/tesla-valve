# tesla-valve


## Description
This repository was created as final project for the class Computer Solutions of Continuum Physics Problems taught in the summer semester 2023/2024. It implements a numerical simulation of a flow through a [Tesla valve](https://en.wikipedia.org/wiki/Tesla_valve).

## Installation
Required packages:
 * [Firedrake] (https://www.firedrakeproject.org/index.html): numerical solution of PDE's,
 * [Netgen] (https://ngsolve.org/): mesh generation, it is most conveniently installed together with Firedrake by providing the [_-netgen_ flag] (https://www.firedrakeproject.org/demos/netgen_mesh.py.html),
 * [NumPy] (https://numpy.org/): fundamental package for scientific computing and working with arrays.

 Visualisation of the results:
 * [ParaView] (https://www.paraview.org/).

## Usage
* `netgen_mesh.py`: This script is used for generation of the Tesla-valve-shaped mesh. The user can adjust: number of lobes (turns) in the valve, mesh size, etc.
* `solve-stokes-mixed.py` is used for solving a time-independent Stokes flow through the Tesla valve using the mixed formulation of the problem (shear stress is treated as a third unknown in addition to the velocity and pressure).
* `solve-navier-stokes.py` produces a time-dependent solution of a flow through the valve. 
* `tesla-valve-drawings.pdf`: Drawings used for creating `netgen_mesh.py` (they provide a visual representation of the code).

## Acknowledgment
The shape of the Tesla valve was sourced from [this webpage](https://www.flickr.com/photos/blightdesign/33571794594/in/photostream/).

During creation of this project, I used and adapted two scripts provided by Patrick Farrell: `steady.py` and `sudden-expansion.py`. Both of them are available in this repository. I also adapted one script from Jaroslav Hron, which was presented during the class. It is available in this repository as `ns_cylinder.py`.

To translate `ns_cylinder.py` from FEniCS to Firedrake, explain certain formulations and debugging, I relied on ChatGPT (version GPT-4o).

## Project status
Work in progress.
