# NOTE: This script can be modified for different atomic structures, 
# units, etc. See in.elastic for more info.
#

# Define the finite deformation size. Try several values of this
# variable to verify that results do not depend on it.
variable up equal 1.0e-2
 
# Define the amount of random jiggle for atoms
# This prevents atoms from staying on saddle points
variable atomjiggle equal 1.0e-5

# Uncomment one of these blocks, depending on what units
# you are using in LAMMPS and for output

# metal units, elastic constants in eV/A^3
#units		metal
#variable cfac equal 6.2414e-7
#variable cunits string eV/A^3

# metal units, elastic constants in GPa
units		metal
variable cfac equal 1.0e-4
variable cunits string GPa

# real units, elastic constants in GPa
#units		real
#variable cfac equal 1.01325e-4
#variable cunits string GPa

# Define minimization parameters
variable etol equal 1e-7
variable ftol equal 1.0e-10
variable maxiter equal 1000000
variable maxeval equal 1000000000
variable dmax equal 1.0e-2

# generate the box and atom positions using a diamond lattice
#variable a equal 5.43

boundary	p p p
box             tilt large
atom_style              atomic



#lattice         diamond $a
#region		box prism 0 2.0 0 3.0 0 4.0 0.0 0.0 0.0
#create_box	1 box
#create_atoms	1 box

# Need to set mass to something, just to satisfy LAMMPS
#mass 1 1.0e-20

read_data               structure.in
mass                    1 72.64
mass                    2 121.760
mass                    3 127.60

