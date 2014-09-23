"""Crystals module.

Data structures for crystal information.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import copy
from ctypes import ArgumentError
import numpy as np
from numpy import cos, sin

#--- Model Imports ---
import crystal_calc
import numpy_utils
from numpy_utils import column, vector_length, rotation_matrix
import ubmatrixreader
import utils

#--- Traits Imports ---
from traits.api import HasTraits,Int,Float,Str,String,Property,Bool, List, Tuple, Array, Enum


# ===========================================================================================
class Crystal(HasTraits):
    """The Crystal class holds information about a crystal
    that a user is using/planning to use in an experiment.
    """
    name = Str("name")
    description = Str("description")

    # For manually specifying the lattice direction lengths
    lattice_lengths_arr = Array( shape=(1,3), dtype=np.float)
    lattice_angles_deg_arr = Array( shape=(1,3), dtype=np.float)
    #Can the lattice exist given these values?
    valid_parameters_yesno = Str("Yes")

    #For manual UB matrix
    #Sample mounting angles in DEGREES!
    sample_mount_phi = Float( 0.)
    sample_mount_chi = Float( 0.)
    sample_mount_omega = Float( 0.)

    #List of options for the point group
    point_group_name_list = ['1']

    #List of options for the reflection condition group
    reflection_condition_name_list = ['Primitive']

    #Resulting or input UB matrix
    ub_matrix = Array( shape=(3,3), dtype=np.float)

    #To show the source of the ub matrix
    ub_matrix_is_from = String("manually generated")

    #Sample mounting orientation matrix
    u_matrix = Array( shape=(3,3), dtype=np.float)

    #Its reciprocal lattice vectors. Uses the physics definition, including the 2pi factor.
    recip_a = Array( shape=(3,), dtype=np.float)
    recip_b = Array( shape=(3,), dtype=np.float)
    recip_c = Array( shape=(3,), dtype=np.float)

    # Real crystal lattice vectors, in XYZ coordinates.
    a = Array( shape=(1,3), dtype=np.float)
    b = Array( shape=(1,3), dtype=np.float)
    c = Array( shape=(1,3), dtype=np.float)

    #Same info, as a column-wise a,b,c matrix
    reciprocal_lattice = Array( shape=(3,3), dtype=np.float)

    #For loading the UB matrix
    ub_matrix_last_filename = Str

    #The hkl indices of interest. A list of (h,k,l) tuples.
    important_hkl = List

    #========================================================================================================
    def __init__(self, name, description=""):
        """Constructor.

        Parameters:
            name and description: string.

        """
        self.name = name
        self.description = description
        #Default values
        self.lattice_lengths_arr = np.array([[5.0, 5.0, 5.0]])
        self.lattice_angles_deg_arr = np.array([[90., 90., 90.]])
        #Generate the UB matrix with the default settings
        self.make_ub_matrix()
        #Make sure the reciprocal is calculated
        self.calculate_reciprocal()
        #Point Group symmetry of the crystal
        self.point_group_name_list = get_point_group_names(long_name=True)
        self.add_trait("point_group_name", Enum( get_point_group_names(long_name=True), value="1" ) )
        # Reflection conditions
        self.reflection_condition_name_list = get_reflection_condition_names()
        self.add_trait("reflection_condition_name", Enum( get_reflection_condition_names(), value="Primitive" ) )
         
        #Pick the first long name
        self.point_group_name = get_point_group_names(long_name=True)[0]

    #========================================================================================================
    def __eq__(self, other):
        return utils.equal_objects(self, other)

    #--- lattice_lengths ----
    def get_lattice_lengths(self):
        """Return the lattice lengths, in angstroms, as a tuple."""
        return tuple(self.lattice_lengths_arr.flatten() )
    def set_lattice_lengths(self, value):
        """Setter for lattice_lengths"""
        self.lattice_lengths_arr = np.array(value).reshape( (1,3) )
    lattice_lengths = property(get_lattice_lengths, set_lattice_lengths)

    #---- lattice_angles -----
    def get_lattice_angles(self):
        """Return the lattice angles, in radians, as a tuple."""
        return tuple( np.deg2rad( self.lattice_angles_deg_arr.flatten() ) )
    def set_lattice_angles(self, value):
        """"Setter for lattice_angles. value = tuple of radian angles."""
        self.lattice_angles_deg_arr = np.rad2deg( np.array(value).reshape( (1,3) ) )
    lattice_angles = property(get_lattice_angles, set_lattice_angles)

    #---- lattice_angles_deg -----
    def get_lattice_angles_deg(self):
        """Return the lattice angles, in degrees, as a tuple."""
        return tuple( self.lattice_angles_deg_arr.flatten() )
    def set_lattice_angles_deg(self, value):
        """"Setter for lattice_angles_deg. value = tuple of degree angles."""
        self.lattice_angles_deg_arr = np.array(value).reshape( (1,3) )
    lattice_angles_deg = property(get_lattice_angles_deg, set_lattice_angles_deg)

    #--------------------------------------------------------------------
    def get_point_group(self):
        """Return the PointGroup object for this crystal."""
        return get_point_group_from_long_name(self.point_group_name)
    
    #--------------------------------------------------------------------
    def get_reflection_conditions(self):
        """Return a list of all ReflectionCondition's that apply """
        return [get_refl_cond(self.reflection_condition_name)]

    #--------------------------------------------------------------------
    def is_lattice_valid(self):
        """Checks if the lattice is valid by trying to calculate
        the lattice volume."""
        #Check the lattice parameters.
        (a,b,c, V) = crystal_calc.make_lattice_vectors(self.lattice_lengths, self.lattice_angles)

        #Bad angles make a nan volume
        #   0 or negative volume also is bad
        return not ( np.isnan(V) or (V <= 1e-5) )


    #--------------------------------------------------------------------
    def make_ub_matrix(self):
        """Generate the UB matrix using the settings in the object
        (the sample mounting angles and lattice params)."""
        #Convert the angles
        phi = np.deg2rad(self.sample_mount_phi)
        chi = np.deg2rad(self.sample_mount_chi)
        omega = np.deg2rad(self.sample_mount_omega)

        #Make U matrix
        self.u_matrix = numpy_utils.rotation_matrix(phi, chi, omega)

        #Now the UB matrix
        self.ub_matrix = crystal_calc.make_UB_matrix(self.lattice_lengths, self.lattice_angles, phi, chi, omega)
        self.ub_matrix_is_from = "\nManually generated.\n"

        # and re-calc the real-space a,b,c vectors
        self.calculate_abc()


    #--------------------------------------------------------------------
    def read_ISAW_ubmatrix_file(self, filename, angles):
        """Load a ISAW-produced UB matrix text file into this crystal.

        Parameters:
            filename: text file to load
            angles: list of 3 sample orientation angles, in radians, at
                time data from UB matrix was taken.
        """
        #Check parameters
        if len(angles) != 3:
            raise ValueError("read_ISAW_ubmatrix_file angles parameter needs 3 angles provided!")

        #Load the file
        ret = ubmatrixreader.read_ISAW_ubmatrix_file(filename, False)
        if not ret is None:
            #load went okay
            (lattice_lengths, lattice_angles_deg, ub_matrix) = ret

            #Save here
            self.lattice_lengths = lattice_lengths
            self.lattice_angles_deg = lattice_angles_deg
            #Make the B matrix etc.
            self.calculate_reciprocal()

#            #Move the columns around
#            UB = np.eye(3)
#            UB[:,0] = ub_matrix[:,1]
#            UB[:,1] = ub_matrix[:,2]
#            UB[:,2] = ub_matrix[:,0]
#            B = self.reciprocal_lattice
#            invB = np.linalg.inv(B)
#            U = np.dot(UB, invB)
#            self.u_matrix = U
#            self.ub_matrix = UB
#            assert np.allclose( np.dot(U, B), UB ), "Calculated and read UB matrices are good."

            #Okay, now we need to account for the ISAW ub matrix file
            #   using IPNS conventions:
            # its coordinates are a right-hand coordinate system where
            #  x is the beam direction and z is vertically upward.(IPNS convention)

            #First we find the U matrix
            original_U = self.calculate_u_matrix(ub_matrix, True)

            #Rotate U to account for goniometer angles.
            (phi, chi, omega) = angles
            gon_rot = numpy_utils.rotation_matrix(phi, chi, omega)
            #Invert the rotation matrix - we want to CANCEL out the goniometer rotation.
            gon_rot = np.linalg.inv(gon_rot)
            #Multiplying the matrix like this (goniometer^-1 * old_U) takes out the goniometer effect to it.
            self.u_matrix = np.dot(gon_rot, original_U)

            #Re-create a UB matrix that uses the SNS convention now
            new_ub_matrix = np.dot(self.u_matrix, self.get_B_matrix())
            self.ub_matrix = new_ub_matrix

            #For next time
            self.ub_matrix_last_filename = filename
            angles_deg = np.rad2deg(angles)
            self.ub_matrix_is_from = "ISAW UB matrix file at\n " + filename + "\nmodified by phi, chi, omega of %.1f, %.1f, %.1f" % (angles_deg[0],angles_deg[1],angles_deg[2])

            # and re-calc the real-space a,b,c vectors
            self.calculate_abc()

    #--------------------------------------------------------------------
    def read_LDM_file(self, filename):
        """Load an Lauegen .ldm format file into this crystal.
        
        Lauegen uses the coordinate system of MOSFLM, see:
         http://www.mrc-lmb.cam.ac.uk/harry/mosflm/mosflm_user_guide.html#a3
        for a description.
        
        For Lauegen:
        http://www.ccp4.ac.uk/ccp4bin/viewcvs/laue/doc/lauegen.txt?rev=HEAD&only_with_tag=JC&content-type=text/vnd.viewcvs-markup
        
        ROTATION_AXIS   This defines which 'signed' reciprocal
                        axial direction is the one closest to the
                        crystal rotation axis. It may be +a*,
                        +b*, +c*, -a*, -b* or -c*. The code may
                        be entered or the value selected from the
                        drop down menu.
        
        BEAM_AXIS       This defines which 'signed' reciprocal
                        axial  direction is the one closest to
                        the X-ray beam direction away from the
                        source. It may be +a*, +b*, +c*, -a*, -b*
                        or -c*. The code may be entered or the
                        value selected from the drop down menu.
                        
        PHIX[], PHIY[], PHIZ[]
                         The crystal missetting angles in degrees
                         around the laboratory frame X, Y and Z
                         axes. (see Appendix 4  for coordinate
                         systems)
                         
        MOSFLM coordinates:
            +X = x-ray beam direction
            +Z = crystal rotation axis = vertical?
            +Y = defined by XYZ forming right-handed coordinate system
        
        LDM file format: The second line:
        
        BEAM_AXIS +a* ROTATION_AXIS +c*
        
        This means that:
        
        - Start with the a* vector along the beam axis
           = in the same direction as the neutron movement
           = +Z in our usual convention (e.g. in ISAW, CrystalPlan and Mantid)
           = +X axis in the MOSFLM convention
        - Start with the c* vector pointing towards the rotation axis
           = the vertical axis pointing upwards (+Y in our usual convention).
           = +Z in the MOSFLM convention 
           = This means that c* is in the YZ plane, with the Y component positive.
        - Use right-handed coordinates, so that the b* vector is inferred
            to be somewhere towards the +X direction.
        
        Then from this starting orientation, we apply the rotations e.g.:
        
        PHIX 85.279 PHIY -47.330 PHIZ 41.879
        
        Here I am not sure if the XYZ match our usual convention. If they do, it would be:
        
        * 85 degrees right-handed rotation about the +X (left) axis
        * -47 degrees right-handed rotation about the +Y (vertical upwards) axis
        * 41 degrees right-handed rotation about the +Z (beam direction) axis
        
        Parameters:
            filename: text file to load
        """
        f = open(filename, 'r')
        
        beam_axis = ""
        rotation_axis = ""
        phix = 0
        phiy = 0
        phiz = 0
        
        for line in f:
            line = line.strip()
            if line.startswith("BEAM_AXIS"):
                try:
                    tokens = line.split(" ")
                    beam_axis = tokens[1]
                    rotation_axis = tokens[3]
                except:
                    raise Exception("Error interpreting LDM string '%s'" % line )
            
            if line.startswith("A "):
                try:
                    # Lattice parameters A 33.920 B 34.874 C 43.478 ALPHA 90.0 BETA 90.0 GAMMA 90.0
                    tokens = line.split(" ")
                    # Convert the odd-numbered ones
                    values = [float(tokens[i]) for i in range(1, len(tokens), 2)]
                    
                    # Lengths in angstroms
                    lattice_lengths = tuple(values[0:3])
                    self.set_lattice_lengths(lattice_lengths)
                    # Angles in degrees
                    angles_deg = tuple(values[3:])
                    angles_rad = tuple([np.deg2rad(x) for x in angles_deg])
                    self.set_lattice_angles_deg(angles_deg)
                    self.set_lattice_angles(angles_rad)
                except:
                    raise Exception("Error interpreting LDM string '%s'" % line )
                
            if line.startswith("PHIX "):
                try:
                    # Rotations PHIX 146.449 PHIY 4.792 PHIZ -171.550
                    tokens = line.split(" ")
                    # Convert the odd-numbered ones
                    values = [float(tokens[i]) for i in range(1, len(tokens), 2)]
                    # Rotation angles in radians
                    (phix, phiy, phiz) = np.deg2rad([x for x in values])
                except:
                    raise Exception("Error interpreting LDM string '%s'" % line )
            
        f.close()

        #If all the sample mounting angles are zero, the sample's crystal
        #lattice coordinate system is aligned with the instrument coordinates.
        #The 'a' vector is parallel to x; 'b' is in the XY plane towards +y;
        #'c' goes towards +z.
        
        #Start with the orientation ABOVE.
        #Now let's make a* point towards +X (in LDM) = +Z (in ISAW)
        R1 = rotation_matrix(phi=-np.pi/2, chi=0, omega=0)
        R = R1
        # Now let's make the b* point towards +Y (in LDM) = +X (in ISAW)
        R2 = rotation_matrix(phi=0, chi=-np.pi/2, omega=0)
        R = np.dot(R2, R)
        
        # Now add the ccw rotation along the LDM +X axis, phix
        # = rotation around ISAW +Z axis
        c = cos(phix)
        s = sin(phix)
        R_phix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        R = np.dot(R_phix, R)
        
        # Now add the rotation along the LDM +Y axis, phiy
        # = rotation around ISAW +X axis
        c = cos(phiy)
        s = sin(phiy)
        R_phiy = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        R = np.dot(R_phiy, R)
        
        # Now add the rotation along the LDM +Z axis, phiz
        # = rotation around ISAW +Y axis
        R_phiz = rotation_matrix(phi=phiz, chi=0, omega=0)
        R = np.dot(R_phiy, R)
        
        # This R is our U matrix
        self.u_matrix = R

        #Now the UB matrix
        self.ub_matrix = crystal_calc.make_UB_matrix(self.lattice_lengths, self.lattice_angles, 0,0,0,
                                                     U_matrix=R)
        self.ub_matrix_is_from = "\nFile %s.\n" % filename

        # and re-calc the real-space a,b,c vectors
        self.calculate_abc()


    #--------------------------------------------------------------------
    def read_HFIR_ubmatrix_file(self, filename, lattice_filename):
        """Load a UB matrix text file produced by the HFIR HB3A beamline software
        into this crystal.

        Parameters:
            filename: text file to load with the ub matrix
            lattice_filename: text file to load with the lattice parameters
        """

        #Load the file
        ub_matrix = ubmatrixreader.read_HFIR_ubmatrix_file(filename)
        ret = ubmatrixreader.read_HFIR_lattice_parameters_file(lattice_filename)

        if not ret is None:
            (lattice_lengths, lattice_angles_deg) = ret

            #Save here
            self.lattice_lengths = lattice_lengths
            self.lattice_angles_deg = lattice_angles_deg
            #Make the B matrix etc.
            self.calculate_reciprocal()

            #print "self.reciprocal_lattice", self.reciprocal_lattice

            # First we find the U matrix
            original_U = self.calculate_u_matrix(ub_matrix, True)

            # And we save it - no rotation required
            self.u_matrix = original_U

            #Re-create a UB matrix that uses the SNS convention now
            new_ub_matrix = np.dot(self.u_matrix, self.get_B_matrix())
            self.ub_matrix = new_ub_matrix

            #For next time
            self.ub_matrix_last_filename = filename
            self.ub_matrix_is_from = "HFIR UB matrix file at\n " + filename 

            # and re-calc the real-space a,b,c vectors
            self.calculate_abc()


    #--------------------------------------------------------------------
    def calculate_reciprocal(self):
        """Calculate the reciprocal lattice of this crystal, from the direct
        lattice parameters."""
        (self.recip_a, self.recip_b, self.recip_c) = \
            crystal_calc.make_reciprocal_lattice(self.lattice_lengths, self.lattice_angles)
        #Also make the matrix
        self.reciprocal_lattice = numpy_utils.vectors_to_matrix(self.recip_a, self.recip_b, self.recip_c)

    #--------------------------------------------------------------------
    def calculate_abc(self):
        """Calculate the abc vectors in real space."""
        (a,b,c, V) = crystal_calc.make_lattice_vectors(self.lattice_lengths, self.lattice_angles)
        #Now rotate all these vectors by the U matrix
        self.a = np.dot(self.u_matrix, a).reshape(1,3)
        self.b = np.dot(self.u_matrix, b).reshape(1,3)
        self.c = np.dot(self.u_matrix, c).reshape(1,3)

#        self.a = a.reshape(3,)
#        self.b = b.reshape(3,)
#        self.c = c.reshape(3,)

    #--------------------------------------------------------------------
    def get_u_matrix(self):
        """Return the previously-calculated U matrix."""
        return self.u_matrix
    

    #--------------------------------------------------------------------
    def calculate_u_matrix(self, ub_matrix, convert_from_IPNS):
        """Calculate the sample's U-matrix: the matrix describing the sample mounting orientation.
        So U = UB * (B)^-1

        Parameters:
            ub_matrix: Either: an ISAW-style UB matrix, after transposing and 2*pi.
                Or: a HFIR-style UB matrix, which ALSO needs coordinate transforms

            convert_from_IPNS: bool, set to True if converting an ISAW (or HFIR)
                ub-matrix file. Coordinates will be transformed from IPNS convention to SNS convention

        """
        #The reciprocal_lattice lattice is the same as the B matrix
        B = self.reciprocal_lattice
        #Try to invert it
        try:
            invB = np.linalg.inv(B)
            U = np.dot(ub_matrix, invB)
            #Test that U must be orthonormal.
            U2 = np.dot(U, U.transpose())
            if not np.allclose(U2, np.eye(3), atol=1e-2):
                print "The U matrix must be orthonormal. Instead, we got:\nU*U.transpose()=%s" % U2

            if (convert_from_IPNS):
                #Okay, now let's permute the rows for IPNS->SNS convention
                U_out = 1. * U
                U_out[2] = U[0] #x gets put in z
                U_out[1] = U[2] #z gets put in y
                U_out[0] = U[1] #y gets put in x
                U = U_out
                
            #Do another test
            U2 = np.dot(U, U.transpose())
            #assert np.allclose(U2, np.eye(3), atol=1e-2), "The U matrix must be orthonormal. Instead, we got:\nU*U.transpose()=%s" % U2
            if not np.allclose(U2, np.eye(3), atol=1e-2):
                print "The U matrix must be orthonormal. Instead, we got:\nU*U.transpose()=%s" % U2
            U_det = np.linalg.det(U)
            #print "Its determinant is %s" % U_det
            assert (abs(U_det-1) < 1e-3), "The U matrix must be a proper reflection. Its determinant is %s" % U_det

            return U
        except np.linalg.LinAlgError:
            raise Error("Invalid reciprocal lattice found; B matrix could not be inverted.")

    #--------------------------------------------------------------------
    def get_B_matrix(self):
        """Returns the B matrix."""
        return  self.reciprocal_lattice


#================================================================================
#============================ GENERATORS ========================================
#================================================================================

class Generator():
    """Class holding info about a generator matrix for calculating point group symmetry."""
    def __init__(self, matrix, expected_symmetry):
        """Parameters:
            matrix: the matrix. Can be a nested list.
            expected_symmetry: ignored
        """
        self.matrix = np.matrix(matrix)
        #Find the symmetry
        # The matrix^symmetry must be equal to identity
        i = 1
        while i < 20:
            if np.allclose(self.matrix**i, np.eye(3)):
                break
            i += 1
        assert i < 20, "symmetry of the generator was found."
        self.symmetry = i

    def __neg__(self):
        """Return the generator with the negative of the matrix."""
        #Matrix is inverted and symmetry is doubled
        gen = Generator(-self.matrix, self.symmetry*2)
        gen.name = "-" + self.name
        return gen
    
    def __str__(self):
        return self.matrix.__str__() + " symmetry=%d" % self.symmetry



#List of all generator matrices.
# Name is 2;1,0,0: 1st number is the n-fold rotation number n; next 3 is about which axis they are made
generators = {}

#----------------------------------------------------------------
def make_generators():
    """Make all the (positive) generator matrices.

    Taken from Giacovazzo, Fundamentals of Crystallography, p.43.
    """
    generators["1"] = Generator([ [1,0,0], [0,1,0], [0,0,1] ], 1)
    generators["2;1,0,0"] = Generator([ [1,0,0], [0,-1,0], [0,0,-1] ], 2)
    generators["H2;1,0,0"] = Generator([ [1,-1,0], [0,-1,0], [0,0,-1] ], 2)
    generators["4;1,0,0"] = Generator([ [1,0,0], [0,0,1], [0,-1,0] ], 4)
    generators["2;0,1,0"] = Generator([ [-1,0,0], [0,1,0], [0,0,-1] ], 2)
    generators["H2;0,1,0"] = Generator([ [-1,0,0], [-1,1,0], [0,0,-1] ], 2)
#    generators["3;0,1,0"] = Generator([ [0,0,-1], [0,1,0], [1,0,0] ], 3) #Possibly incorrect? Anyway, its not used

    generators["2;0,0,1"] = Generator([ [-1,0,0], [0,-1,0], [0,0,1] ], 2)
    generators["H3;0,0,1"] = Generator([ [0,-1,0], [1,-1,0], [0,0,1] ], 3)
    generators["4;0,0,1"] = Generator([ [0,-1,0], [1,0,0], [0,0,1] ], 4)
    generators["H6;0,0,1"] = Generator([ [1,-1,0], [1,0,0], [0,0,1] ], 6)
    
    generators["2;1,1,0"] = Generator([ [0,1,0], [1,0,0], [0,0,-1] ], 2)
    generators["2;1,0,1"] = Generator([ [0,0,1], [0,-1,0], [1,0,0] ], 2)
    generators["2;0,1,1"] = Generator([ [-1,0,0], [0,0,1], [0,1,0] ], 2)
    generators["2;1,-1,0"] = Generator([ [0,-1,0], [-1,0,0], [0,0,-1] ], 2)
    generators["2;-1,0,1"] = Generator([ [0,0,-1], [0,-1,0], [-1,0,0] ], 2)
    generators["2;0,1,-1"] = Generator([ [-1,0,0], [0,0,-1], [0,-1,0] ], 2)

    generators["3;1,1,1"] = Generator([ [0,0,1], [1,0,0], [0,1,0] ], 3)
    generators["3;-1,1,1"] = Generator([ [0,-1,0], [0,0,1], [-1,0,0] ], 3)
    generators["3;1,-1,1"] = Generator([ [0,-1,0], [0,0,-1], [1,0,0] ], 3)
    generators["3;1,1,-1"] = Generator([ [0,1,0], [0,0,-1], [-1,0,0] ], 3)

    generators["H2;2,1,0"] = Generator([ [1,0,0], [1,-1,0], [0,0,-1] ], 2)
    generators["H2;1,2,0"] = Generator([ [-1,1,0], [0,1,0], [0,0,-1] ], 2)

    for key in generators.keys():
        generators[key].name = key


#----------------------------------------------------------------
def permutations(iterable, r=None):
    """Return successive r length permutations of elements in the iterable. Replacement for itertools.permutations, which is only present in v.2.6
    If r is not specified or is None, then r defaults to the length of the iterable and all possible full-length permutations are generated.
    Permutations are emitted in lexicographic sort order. So, if the input iterable is sorted, the permutation tuples will be produced in sorted order.
    Elements are treated as unique based on their position, not on their value. So if the input elements are unique, there will be no repeat values in each permutation"""
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = range(n)
    cycles = range(n, n-r, -1)
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return

#----------------------------------------------------------------
def product(*args, **kwds):
    """Cartesian product of input iterables.
    Equivalent to nested for-loops in a generator expression. For example, product(A, B) returns the same as ((x,y) for x in A for y in B)."""
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


#================================================================================
#============================ POINT GROUPS ======================================
#================================================================================
class PointGroup():
    """Class describing the symmetry of a particular point group.
    """

    #----------------------------------------------------------------
    def __init__(self, name, long_name, expected_order, preferred_axis, *args):
        """Constructor.

        Parameters:
            name: string, short name of the point group, e.g. "4mm".
                Use a "-" as a prefix to indicate "bar", e.g. 1-bar = "-1"

            long_name: string, longer description of the point group
            expected_order: order of the point group (aka number of matrices in the
                multiplication table). An error is raised if the calculation
                does not yield this expected number
            preferred_axis: string, 'h', 'k', or 'lkh', the rotation axis to prioritize
                when choosing which peaks are 'primary' ones
            args: one or more 3x3 generator matrices names, as strings, e.g. "2;1,0,0".
                use a "-" to indicate a negative generator, "-2;1,0,0"
        """
        self.name = name
        self.long_name = long_name
        self.preferred_axis = preferred_axis.lower()
        if len(args)==0:
            raise ArgumentError("You must supply at least one generator matrix.")
        self.generators = []
        for name in args:
            if name[0] == "-":
                self.generators.append( -generators[name[1:]] )
            else:
                self.generators.append( generators[name] )
        self.make_table()
        assert len(self.table)==expected_order, "order of the Point Group (%d) matches expected value (%d)?" % (len(self.table), expected_order)

    #----------------------------------------------------------------
    def make_table(self):
        """Generate the full multiplication table based on the generators."""
        #Start with an empty list 
        self.table = [ ]
        # @type gen Generator

        #We make a list of lists of each possible factor of the generators
        gen_factors = []

        for gen in self.generators:
            this_gen_factors = []
            for i in xrange(1,gen.symmetry+1):
                this_gen_factors.append( gen.matrix**i )
            gen_factors.append( this_gen_factors )

        for this_product in product(*gen_factors):
            #Start with identity matrix
            matrix = np.matrix(np.eye(3))
            for m in this_product:
                matrix = matrix * m
            #Is the matrix already in there
            matrix_already_there = False
            for m in self.table:
                if np.allclose(matrix, m):
                    matrix_already_there = True
                    break
            #No? It is unique, add it
            if not matrix_already_there:
                self.table.append(matrix)
            

    #----------------------------------------------------------------
    def make_table_old(self):
        i = [0]*len(gen)
        mult_list = []
        #Each generator can be applied up to gen.symmetry times
        for gen in self.generators:
            for i in xrange(1,gen.symmetry+1):
                mult_list.append( gen.matrix )

        #So we look at applying 1 to X elements from the list
        for num_operations in xrange(1, len(mult_list)+1):
            #But matrix multiplications are not commutative, in general!
            #So now we need to go through every permuation of the order of that list
            # Python, in its wise ways, gives us a function for that
            permu_list = [x for x in permutations(mult_list, num_operations)]
            
            #Now we make it into a set to remove doubled elements
            permu_set = set(permu_list)

            for this_order in permu_set:
                #Start with identity
                matrix = np.matrix(np.eye(3))
                for m in this_order:
                    matrix = matrix * m

                #Is the matrix already in there
                matrix_already_there = False
                for m in self.table:
                    if np.allclose(matrix, m):
                        matrix_already_there = True
                        break
                #No? It is unique, add it
                if not matrix_already_there:
                    self.table.append(matrix)

    #----------------------------------------------------------------
    def __str__(self):
        """Generate a string describing the point group."""
        s = "Point group '%s' (%s) has %d generator(s) and is of order %d." % (self.name, self.desc, len(self.generators), len(self.table))
        return s

    #----------------------------------------------------------------
    def table_is_equivalent(self, other):
        """Returns True if the table in 'other' is equivalent to the one in this object."""
        if other is None: return False
        if len(other.table) != len(self.table):
            return False
        for x in other.table:
            found = False
            for y in self.table:
                if np.allclose(x,y):
                    found = True
                    break
            if not found:
                return False
        return True

    #----------------------------------------------------------------
    def get_equivalent_hkl(self, hkl):
        """Return all equivalent HKL values.

        Parameters:
            hkl: tuple or vector with the hkl values of any peak.

        Return:
            a list of (h,k,l) tuples.
        """
        results = []
        hkl = column(hkl)
        for matrix in self.table:
            other_hkl = np.array(np.dot(matrix, hkl)).flatten()
            #Make into a tuple of ints.
            results.append(tuple([int(x) for x in other_hkl]))
        return results


# Do we make all 32 point groups (True)? Or only the 11 with a center of inversion (False)?
make_all_point_groups = True

#List of all point groups
point_groups = []

def unique(x):
    matrix_already_there = False
    for m in table:
        if np.allclose(x, m):
            matrix_already_there = True
            break
    #No? It is unique, add it
    if not matrix_already_there:
        print "..adding it!"
        table.append(x)
        print table
    else:
        print "already there!"
    

#----------------------------------------------------------------
#----------------------------------------------------------------
def make_all_point_groups():
    """Hard-coded function that generates all the standard point groups.

    Generator matrices taken from Giacovazzo, Fundamentals of Crystallography, p.46.
    Generator choice from IUCR International Tables for Crystallography, Vol A, ch.8.3, table 8.3.5.1, p.737.
    """

    #11 Laue classes
    point_groups.append( PointGroup("-1", "-1 (Triclinic)", 2, 'lkh', "-1") )
    point_groups.append( PointGroup("1 2/m 1", "1 2/m 1 (Monoclinic, unique axis b)", 4, 'kl', "2;0,1,0", "-1" ))
    point_groups.append( PointGroup("1 1 2/m", "1 1 2/m (Monoclinic, unique axis c)", 4, 'kl', "2;0,0,1", "-1" ))
    point_groups.append( PointGroup("mmm", "mmm (Orthorombic)", 8, 'lkh', "-2;1,0,0", "-2;0,1,0", "-2;0,0,1" ))
    point_groups.append( PointGroup("4/m", "4/m (Tetragonal)", 8, 'lkh', "2;0,0,1", "4;0,0,1", "-1" ))
    point_groups.append( PointGroup("4/mmm", "4/mmm (Tetragonal)", 16, 'lkh', "2;0,0,1", "4;0,0,1", "2;0,1,0", "-1" ))
    point_groups.append( PointGroup("-3", "-3 (Trigonal - Hexagonal)", 6, '3kh', "-H3;0,0,1"))
    point_groups.append( PointGroup("-3m1", "-3m1 (Trigonal - Rhombohedral)", 12, '3kh', "H3;0,0,1", "-2;1,1,0", "-1"))
    point_groups.append( PointGroup("-31m", "-31m (Trigonal - Rhombohedral)", 12, '3kh', "H3;0,0,1", "-2;1,-1,0", "-1"))
    point_groups.append( PointGroup("6/m", "6/m (Hexagonal)", 12, 'lkh', "H6;0,0,1", "-2;0,0,1" ))
    point_groups.append( PointGroup("6/mmm", "6/mmm (Hexagonal)", 24, 'lkh', "H6;0,0,1", "2;1,1,0", "-1" ))
    point_groups.append( PointGroup("m-3", "m-3 (Cubic)", 24, 'lkh', "2;0,0,1", "2;0,1,0", "3;1,1,1", "-1" ))
    point_groups.append( PointGroup("m-3m", "m-3m (Cubic)", 48, 'lkh',"2;0,0,1", "2;0,1,0", "3;1,1,1", "2;1,1,0", "-1" ))

    if make_all_point_groups:
        #The following point groups do not have an inversion center,
        #   and therefore are not meaningful in diffraction experiments
        point_groups.append( PointGroup("1", "1 (Triclinic)", 1, 'lkh', "1") )
        
        point_groups.append( PointGroup("2", "2 (Monoclinic)", 2, 'k', "2;0,1,0"))
        point_groups.append( PointGroup("m", "m (Monoclinic)", 2, 'k', "-2;0,0,1"))
        
        point_groups.append( PointGroup("222", "222 (Orthorombic)", 4, 'lkh', "2;1,0,0", "2;0,1,0" ))
        point_groups.append( PointGroup("mm2", "mm2 (Orthorombic)", 4, 'lkh', "-2;1,0,0", "2;0,0,1" ))

        point_groups.append( PointGroup("4", "4 (Tetragonal)", 4, 'lkh', "4;0,0,1"))
        point_groups.append( PointGroup("-4", "-4 (Tetragonal)", 4, 'lkh', "-4;0,0,1"))
        point_groups.append( PointGroup("422", "422 (Tetragonal)", 8, 'lkh', "2;1,0,0", "4;0,0,1", "2;1,-1,0" ))
        point_groups.append( PointGroup("4mm", "4mm (Tetragonal)", 8, 'lkh', "-2;1,0,0", "4;0,0,1", "-2;1,-1,0" ))
        point_groups.append( PointGroup("-42m", "-42m (Tetragonal)", 8, 'lkh', "2;0,0,1", "-4;0,0,1", "2;0,1,0" ))
        point_groups.append( PointGroup("-4m2", "-4m2 (Tetragonal)", 8, 'lkh', "2;0,0,1", "-4;0,0,1", "-2;0,1,0" ))
        
        point_groups.append( PointGroup("3", "3 (Trigonal)", 3, '3kh', "H3;0,0,1"))
        point_groups.append( PointGroup("321", "321 (Trigonal)", 6, '3kh', "H3;0,0,1", "2;1,1,0" )) 
        point_groups.append( PointGroup("312", "312 (Trigonal)", 6, '3kh', "H3;0,0,1", "2;1,-1,0" ))
        point_groups.append( PointGroup("3m1", "3m1 (Trigonal)", 6, '3kh', "H3;0,0,1", "-2;1,1,0" )) 
        point_groups.append( PointGroup("31m", "31m (Trigonal)", 6, '3kh', "H3;0,0,1", "-2;1,-1,0" )) 
        
        point_groups.append( PointGroup("6", "6 (Hexagonal)", 6, 'lkh', "H6;0,0,1"))
        point_groups.append( PointGroup("-6", "-6 (Hexagonal)", 6, 'lkh', "-H6;0,0,1"))
        point_groups.append( PointGroup("622", "622 (Hexagonal)", 12, 'lkh', "H3;0,0,1", "2;0,0,1", "2;1,1,0")) #"2;1,0,0", "2;1,-1,0" ))
        point_groups.append( PointGroup("6mm", "6mm (Hexagonal)", 12, 'lkh', "H3;0,0,1", "2;0,0,1", "-2;1,1,0")) #"-2;1,0,0", "-2;1,-1,0" ))
        point_groups.append( PointGroup("-6m2", "-6m2 (Hexagonal)", 12, 'lkh', "H3;0,0,1", "-2;0,0,1", "-2;1,1,0")) #  "-2;1,0,0", "2;1,-1,0" ))
        point_groups.append( PointGroup("-62m", "-62m (Hexagonal)", 12, 'lkh', "H3;0,0,1", "-2;0,0,1", "2;1,1,0")) #  "-2;1,0,0", "2;1,-1,0" ))
        
        point_groups.append( PointGroup("23", "23 (Cubic)", 12, 'lhk', "2;0,0,1", "2;0,1,0", "3;1,1,1")) #"3;1,1,1", "2;0,0,1" ))
        point_groups.append( PointGroup("432", "432 (Cubic)", 24, 'lkh', "2;0,0,1", "2;0,1,0", "3;1,1,1", "2;1,1,0")) #"3;1,1,1", "2;1,1,0" ))
        point_groups.append( PointGroup("-43m", "-43m (Cubic)", 24, 'lkh', "2;0,0,1", "2;0,1,0", "3;1,1,1", "-2;1,-1,0")) #"3;1,1,1", "-2;1,1,0" ))


#================================================================================
def get_point_group_names(long_name=False):
    """Returns a list of all the point groups' names.
    
    Parameters:
        long_name: True to return the long name, False to return the short name
    """
    if long_name:
        return [pg.long_name for pg in point_groups]
    else:
        return [pg.name for pg in point_groups]

#================================================================================
def get_point_group_from_name(name):
    """Returns a point group given its short name."""
    for pg in point_groups:
        if pg.name == name:
            return pg
    return None

#================================================================================
def get_point_group_from_long_name(long_name):
    """Returns a point group given its long name."""
    for pg in point_groups:
        if pg.long_name == long_name:
            return pg
    return None

#================================================================================
def _initialize():
    """Initialize the calculations necessary for this module."""
    make_generators()
    make_all_point_groups()

if len(point_groups)==0:
    _initialize()





#================================================================================
#============================ REFLECTION CONDITIONS =============================
#================================================================================
class ReflectionCondition:
    """Class describing reflection conditions, e.g.
        h + k = 2n for (hkl),
    and utility functions for filtering all HKL into
    just those that match the condition."""
    
    def __init__(self, name, applies_to, reflection_condition):
        """Constructor:
        Parameters:
            name: name of the reflection condition
            applies_to: python string that evaluates to a vector of [true] if the
                given h,k,l are one that applies to. (h k l are vectors)
            reflection_condition: python string that evaluates to a vector of [true]
                if the given h,k,l reflection WILL be present in diffraction.
        """
        self.name = name
        self.applies_to = applies_to
        self.reflection_condition = reflection_condition
        
    # ---------------------------------------------------------------------------
    def applies(self, h, k, l):
        """Does this reflection condition apply to these hkl?
        Parameters:
            h,k,l : vectors of h, k, and l.
        Returns:
            vector of True/False.
        """
        # Make sure inputs are flat vectors
        h = np.array(h).flatten()
        k = np.array(k).flatten()
        l = np.array(l).flatten()
        # Evaluate the python bit
        ret = eval(self.applies_to)
        return ret
    
    
    # ---------------------------------------------------------------------------
    def reflection_visible(self, h, k, l):
        """Will this reflection be visible given the reflection conditions?
        This takes into account whether or not the condition applies, e.g.
        if it does not apply, then it returns True; if it does apply,
        it evaluates the condition to determine truth 
        
        Parameters:
            h,k,l : vectors of h, k, and l.
        Returns:
            vector of True/False.
        """
        # Make sure inputs are flat vectors
        h = np.array(h).flatten()
        k = np.array(k).flatten()
        l = np.array(l).flatten()
        # Get the applies vector
        apply = self.applies(h, k, l)
        # Evaluate the python bit
        ret = eval(self.reflection_condition)
        # Anything where it did not apply becomes "true"
        ret = ret | ~apply
        return ret
            
    # ---------------------------------------------------------------------------
    def reflection_visible_matrix(self, hkl):
        """Do reflection_visible() on a 3xN array of HKL
        
        Parameters:
            hkl : ndarray of 3xN values of h, k, and l.
        Returns:
            vector of True/False.
        """        
        h = hkl[0,:]
        k = hkl[1,:]
        l = hkl[2,:]
        return self.reflection_visible(h, k, l)

""" List of ReflectionCondition object """        
refl_conds = []


#================================================================================
def make_all_reflection_conditions():
    """ Generate all the reflection conditions used in the program """
    def add(reflcond):
        refl_conds.append(reflcond);
        
    add( ReflectionCondition("Primitive", applies_to="h==h", reflection_condition="h==h"))
    add( ReflectionCondition("C centred", applies_to="h==h", reflection_condition="((h+k)%2)==0"))
    add( ReflectionCondition("A centred", applies_to="h==h", reflection_condition="((k+l)%2)==0"))
    add( ReflectionCondition("B centred", applies_to="h==h", reflection_condition="((h+l)%2)==0"))
    add( ReflectionCondition("I centred", applies_to="h==h", reflection_condition="((h+k+l)%2)==0"))
    
    # condition: h+k, h+l and k+l==2n; or all even, or all odd.
    add( ReflectionCondition("F centred", applies_to="h==h", 
                             reflection_condition="((((h+k)%2)==0) & (((h+l)%2)==0) & (((k+l)%2)==0)) | ((h%2==0) & (k%2==0) & (l%2==0)) | ((h%2==1) & (k%2==1) & (l%2==1))"))
    
    add( ReflectionCondition("R centred, obverse", applies_to="h==h", reflection_condition="((-h+k+l)%3)==0"))
    add( ReflectionCondition("R centred, reverse", applies_to="h==h", reflection_condition="((h-k+l)%3)==0"))
    add( ReflectionCondition("Hexagonally centred, reverse", applies_to="h==h", reflection_condition="((h-k)%3)==0"))

#================================================================================
def get_reflection_condition_names():
    """Returns a list of all the ReflectionCondition' names. """
    return [rc.name for rc in refl_conds]

#================================================================================
def get_refl_cond(name):
    """Returns the ReflectionCondition named. """
    for rc in refl_conds:
        if rc.name == name:
            return rc
    return None

if len(refl_conds)==0:
    make_all_reflection_conditions()
        

#================================================================================
#============================ UNIT TESTING ======================================
#================================================================================
import unittest
from numpy import pi

def matches(a,b):
    """Match a list of 0,1 to an array of True/False"""
    return np.all( (np.array(a)==1) == b)

#==================================================================
class TestReflectionCondition(unittest.TestCase):
    """Unit test for the Crystal class."""

    def test_CFC(self):
        rc = get_refl_cond("C centred")
        h = [0,0,0,1,1,1,2,2,2]
        k = [0,1,2,0,1,2,0,1,2]
        l = [0,1,3,4,5,6,7,8,9]
        v = [1,0,1,0,1,0,1,0,1] # 1 == this will be valid
        res_vis = rc.reflection_visible(h, k, l)
        assert matches(v, res_vis)

        # Same but with the matrix call        
        hkl = np.array([h,k,l])
        res_vis2 = rc.reflection_visible_matrix(hkl)
        assert matches(v, res_vis2)
        
    def test_All_FC(self):
        rc = get_refl_cond("F centred")
        h = [0,1,0,1,1]
        k = [0,1,0,3,2]
        l = [0,1,1,1,3]
        v = [1,1,0,1,0] # 1 == this will be valid
        res_vis = rc.reflection_visible(h, k, l)
        assert matches(v, res_vis)
        
        
        
    def test_applies(self):
        rc = ReflectionCondition("test", applies_to="h==0", reflection_condition="((k+l)%2)==0")
        h = [0,0,0,0,1,1]
        k = [0,1,0,1,0,1]
        l = [0,0,1,1,0,1]
        a = [1,1,1,1,0,0] # applies
        v = [1,0,0,1,1,1] # 1 == this will be valid
        
        res_app = rc.applies(h, k, l)
        assert matches(a, res_app)
         
        res_vis = rc.reflection_visible(h, k, l)
        assert matches(v, res_vis)

        


#==================================================================
class TestCrystal(unittest.TestCase):
    """Unit test for the Crystal class."""
    def setUp(self):
        self.c = Crystal("my name")

    def test_constructor(self):
        """Crystal.__init__()"""
        c = self.c
        assert c.lattice_angles_deg == (90.,90.,90.), "Lattice angles default values"
        #assert c.lattice_lengths == (10.,10.,10.), "Lattice lengths default values"

    def test_calculate_reciprocal(self):
        """Crystal.calculate_reciprocal()"""
        c = self.c
        c.lattice_lengths = (10,2,3)
        c.calculate_reciprocal()
        assert np.allclose(c.recip_a, (0.1*2*pi,0,0)), "Reciprocal vector a"
        assert np.allclose(c.recip_b, (0,0.5*2*pi,0)), "Reciprocal vector b"
        assert np.allclose(c.recip_c, (0,0,1./3*2*pi)), "Reciprocal vector c"

    def test_make_ub_matrix(self):
        """Crystal.make_ub_matrix()"""
        c = self.c
        c.lattice_lengths = (1,1,1)
        c.calculate_reciprocal()
        c.make_ub_matrix()
        assert np.allclose(c.ub_matrix, np.identity(3)*2*pi), "Default UB matrix is identity*2*pi"
        #Now test get_u

    def calc_ub(self, phi,chi,omega):
        self.c.sample_mount_phi = phi
        self.c.sample_mount_chi = chi
        self.c.sample_mount_omega = omega
        self.c.make_ub_matrix()
        return numpy_utils.rotation_matrix( np.deg2rad(phi), np.deg2rad(chi), np.deg2rad(omega))

    def test_get_u_matrix(self):
        c = self.c
        c.lattice_lengths = (1,1,1)
        c.calculate_reciprocal()
        M = self.calc_ub(15,35,-80)
        U = c.get_u_matrix()
        assert np.linalg.det(U) == 1, "get_u_matrix: det(U) == 1. Instead we got %s" % (np.linalg.det(U))
        assert np.allclose(M, U), "get_u_matrix for some angles 1."
        M = self.calc_ub(10,-30,+345)
        assert np.allclose(M, c.get_u_matrix()), "get_u_matrix for some angles 2."

    def test_properties(self):
        """Crystal properties"""
        c = self.c
        c.lattice_lengths = (1,2,3)
        assert c.lattice_lengths == (1.,2.,3.), "Lattice lengths setter worked"
        c.lattice_angles_deg = (45,60,90)
        assert c.lattice_angles_deg == (45.,60.,90.), "Lattice angles (degree) setter worked"
        c.lattice_angles = (np.pi/4,np.pi/3,np.pi/2)
        assert np.allclose(c.lattice_angles, (np.pi/4,np.pi/3,np.pi/2)), "Lattice angles (rad) setter worked"
        assert np.allclose(c.lattice_angles_deg, (45.,60.,90.)), "Lattice angles (rad) setter worked"

    def DONT_test_read_ubmatrix_file(self):
        """Crystal.read_ISAW_ubmatrix_file
        """
        c = self.c
        c.read_ISAW_ubmatrix_file("data/sampleubMatrix.txt")
        #Here is the (transposed) UB matrix from that file.
        ub = np.fromstring(""" -0.017621 -0.035104 -0.037863
                             -0.018495 -0.032401  0.038647 
                             -0.133741  0.071505 -0.004055""", sep=" ")
        #2 pi factor is NOT included in the file given.
        ub = ub.reshape( 3,3 ).transpose() * 2*pi
        assert np.allclose(c.ub_matrix, ub), "UB matrices match. I read %s" % (ub)

    def test_ub_matrix_and_recip_lattice(self):
        #@type c Crystal
        c = self.c
        c.read_ISAW_ubmatrix_file("data/natrolite_807_ev.mat", [0,0,0])
        UB = c.ub_matrix
        print "UB matrix loaded (including 2pi) is:\n", UB

        print "reciprocal lattice (B) found using the direct lattice params:\n", c.reciprocal_lattice
        g_star = np.dot(c.reciprocal_lattice.transpose(), c.reciprocal_lattice.transpose())
        print "a*.a* is", vector_length(c.recip_a)**2
        print "b*.b* is", vector_length(c.recip_b)**2
        print "c*.c* is", vector_length(c.recip_c)**2
        g_star3 = np.dot(UB.transpose(), UB)
        print "G* found using the direct lattice params:\n", g_star
        print "G* found using UB.transpose() * UB:\n", g_star3
        
        #Check the U matrix
        U = c.get_u_matrix()
        assert np.allclose(np.linalg.det(U), 1), "get_u_matrix: det(U) == 1. Instead we got %s" % (np.linalg.det(U))


    def test_read_is_lattice_valid(self):
        """Crystal.is_lattice_valid"""
        c = self.c
        assert c.is_lattice_valid(), "default lattice is valid."
        c.lattice_angles_deg = ( 5, 3, 90)
        assert not c.is_lattice_valid(), "Bad angles = invalid lattice."
        c.lattice_angles_deg = ( 45, 45, 45)
        assert c.is_lattice_valid(), "Okay angles = valid lattice."
        c.lattice_lengths = ( 1.0, 0, 0)
        assert not c.is_lattice_valid(), "Bad lengths = invalid lattice."

    def test_point_groups_equivalency(self):
        pg1 = PointGroup("4/m", "4/m (Tetragonal)", 8, 'lkh', "4;0,0,1", "-2;0,0,1" )
        pg2 = PointGroup("4/m", "4/m (Tetragonal)", 8, 'lkh', "4;0,0,1", "-1" )
        assert pg1.table_is_equivalent(pg2), "4/m generated 2 ways"

        pg2 = PointGroup("mmm", "mmm (Orthorombic)", 8, 'lkh', "-2;1,0,0", "-2;0,1,0", "-2;0,0,1" )
        assert not pg1.table_is_equivalent(pg2), "4/m isn't equivalent to mmm"

        pg1 = PointGroup("6/m", "6/m (Hexagonal)", 12, 'lkh', "H6;0,0,1", "-2;0,0,1" )
        pg2 = PointGroup("6/m", "6/m (Hexagonal)", 12, 'lkh', "H6;0,0,1", "-1")
        assert pg1.table_is_equivalent(pg2), "6/m generated 2 ways"

    def check_point_group(self, pg_name, hkl, hkl_expected):
        message = "point group %s and hkl %s" % (pg_name, hkl)

        #@type pg PointGroup
        #We get the point group
        pg = get_point_group_from_name(pg_name)
        #Find the list of HKL
        found = pg.get_equivalent_hkl(hkl)
        #print message, "; found ", found

        assert len(found)==len(hkl_expected), "correct # of results for %s. We wanted %d results but got %d" % (message, len(hkl_expected), len(found))
        for hkl_wanted in hkl_expected:
            assert hkl_wanted in found, "hkl %s was found in the list of results for %s.\nfound=%s" % (hkl_wanted, message,found)

    def test_point_groups(self):
        """Exhaustive and exhausting test each of the 11 Laue classes/32 point groups equivalent hkls"""
        self.check_point_group("-1", (1,2,3), [(1,2,3),(-1,-2,-3)] )
        self.check_point_group("1 2/m 1", (1,2,3), [(1,2,3), (-1,-2,-3), (-1,2,-3), (1,-2,3)  ])
        self.check_point_group("1 1 2/m", (1,2,3), [(1,2,3), (-1,-2,3), (-1,-2,-3), (1,2,-3)  ])
        self.check_point_group("mmm", (1,2,3), [(1,2,3),(-1,-2,3), (-1,2,-3), (1,-2,-3), (-1,-2,-3), (1,2,-3), (1,-2,3), (-1,2,3)] )
        self.check_point_group("4/m", (1,2,3), [(1,2,3),(-1,-2,3), (-2,1,3), (2,-1,3), (-1,-2,-3), (1,2,-3), (2,-1,-3), (-2,1,-3)] )
        self.check_point_group("4/mmm", (1,2,3), [(1,2,3),(-1,-2,3), (-2,1,3), (2,-1,3), (-1,2,-3), (1,-2,-3), (2,1,-3), (-2,-1,-3), (-1,-2,-3), (1,2,-3), (2,-1,-3), (-2,1,-3), (1,-2,3), (-1,2,3),(-2,-1,3), (2,1,3)] )
        self.check_point_group("-3", (1,2,3), [(1,2,3),(-2,1-2,3), (-1+2,-1,3), (-1,-2,-3), (2,-1+2,-3), (1-2,1,-3)] )
        self.check_point_group("-31m", (1,2,3), [(1,2,3),(-2,1-2,3),(-1+2,-1,3),(-2,-1,-3),(-1+2,2,-3),(1,1-2,-3),(-1,-2,-3),(2,-1+2,-3),(1-2,1,-3),(2,1,3),(1-2,-2,3),(-1,-1+2,3)])
        self.check_point_group("-3m1", (1,2,3), [(1,2,3),(-2,1-2,3),(-1+2,-1,3),(2,1,-3),(1-2,-2,-3),(-1,-1+2,-3),(-1,-2,-3),(2,-1+2,-3),(1-2,1,-3),(-2,-1,3),(-1+2,2,3),(1,1-2,3)] )
        self.check_point_group("6/m", (1,2,3), [(1,2,3),(-2,1-2,3),(-1+2,-1,3),(-1,-2,3),(2,-1+2,3),(1-2,1,3),(-1,-2,-3),(2,-1+2,-3),(1-2,1,-3),(1,2,-3),(-2,1-2,-3),(-1+2,-1,-3)] )
        self.check_point_group("6/mmm", (1,2,3), [(1,2,3),(-2,1-2,3),(-1+2,-1,3),(-1,-2,3),(2,-1+2,3),(1-2,1,3),(2,1,-3),(1-2,-2,-3),(-1,-1+2,-3),(-2,-1,-3),(-1+2,2,-3),(1,1-2,-3),(-1,-2,-3),(2,-1+2,-3),(1-2,1,-3),(1,2,-3),(-2,1-2,-3),(-1+2,-1,-3),(-2,-1,3),(-1+2,2,3),(1,1-2,3),(2,1,3),(1-2,-2,3),(-1,-1+2,3)])
        self.check_point_group("m-3", (1,2,3), [(1,2,3),(-1,-2,3),(-1,2,-3),(1,-2,-3),(3,1,2),(3,-1,-2),(-3,-1,2),(-3,1,-2),(2,3,1),(-2,3,-1),(2,-3,-1),(-2,-3,1),(-1,-2,-3),(1,2,-3),(1,-2,3),(-1,2,3),(-3,-1,-2),(-3,1,2),(3,1,-2),(3,-1,2),(-2,-3,-1),(2,-3,1),(-2,3,1),(2,3,-1)] )
        self.check_point_group("m-3m", (1,2,3), [(1,2,3),(-1,-2,3),(-1,2,-3),(1,-2,-3),(3,1,2),(3,-1,-2),(-3,-1,2),(-3,1,-2),(2,3,1),(-2,3,-1),(2,-3,-1),(-2,-3,1),(2,1,-3),(-2,-1,-3),(2,-1,3),(-2,1,3),(1,3,-2),(-1,3,2),(-1,-3,-2),(1,-3,2),(3,2,-1),(3,-2,1),(-3,2,1),(-3,-2,-1),(-1,-2,-3),(1,2,-3),(1,-2,3),(-1,2,3),(-3,-1,-2),(-3,1,2),(3,1,-2),(3,-1,2),(-2,-3,-1),(2,-3,1),(-2,3,1),(2,3,-1),(-2,-1,3),(2,1,3),(-2,1,-3),(2,-1,-3),(-1,-3,2),(1,-3,-2),(1,3,2),(-1,3,-2),(-3,-2,1),(-3,2,-1),(3,-2,-1),(3,2,1)] )
        
        # Check the other ones!
        if make_all_point_groups:
            print "All point groups"
            self.check_point_group("1", (1,2,3), [(1,2,3)] )
            self.check_point_group("2", (1,2,3), [(1,2,3),(-1,2,-3)] )
            self.check_point_group("m", (1,2,3), [(1,2,3),(1,2,-3)] )
            self.check_point_group("222", (1,2,3), [(1,2,3),(-1,-2,3),(-1,2,-3),(1,-2,-3),] )
            self.check_point_group("mm2", (1,2,3), [(1,2,3),(-1,-2,3),(1,-2,3),(-1,2,3),] )
            self.check_point_group("4", (1,2,3), [(1,2,3),(-1,-2,3),(-2,1,3),(2,-1,3),] )
            self.check_point_group("-4", (1,2,3), [(1,2,3),(-1,-2,3),(2,-1,-3),(-2,1,-3),] )
            self.check_point_group("422", (1,2,3), [(1,2,3),(-1,-2,3),(-2,1,3),(2,-1,3),  (-1,2,-3),(1,-2,-3),(2,1,-3),(-2,-1,-3),] )
            self.check_point_group("4mm", (1,2,3), [(1,2,3),(-1,-2,3),(-2,1,3),(2,-1,3),  (1,-2,3),(-1,2,3),(-2,-1,3),(2,1,3),] )
            self.check_point_group("4mm", (1,2,3), [(1,2,3),(-1,-2,3),(-2,1,3),(2,-1,3),  (1,-2,3),(-1,2,3),(-2,-1,3),(2,1,3),] )
            self.check_point_group("-42m", (1,2,3), [(1,2,3),(-1,-2,3),(2,-1,-3),(-2,1,-3),  (-1,2,-3),(1,-2,-3),(-2,-1,3),(2,1,3),] )
            self.check_point_group("-4m2", (1,2,3), [(1,2,3),(-1,-2,3),(2,-1,-3),(-2,1,-3),  (1,-2,3),(-1,2,3),(2,1,-3),(-2,-1,-3),] )


    def test_read_HFIR_ubmatrix_file(self):
        #@type c Crystal
        c = self.c
        c.read_HFIR_ubmatrix_file("data/HFIR_UBmatrix.dat", "data/HFIR_lattice.dat")
        UB = c.ub_matrix
        print "UB matrix loaded (including 2pi) is:\n", UB

    def test_read_LDM_file(self):
        #@type c Crystal
        c = self.c
        c.read_LDM_file("data/LADI_fd14.ldm")
        UB = c.ub_matrix
        print "UB matrix loaded (including 2pi) is:\n", UB


#---------------------------------------------------------------------
if __name__ == "__main__":
#    unittest.main()

    tst = TestCrystal('test_read_LDM_file')
    tst.setUp()
    tst.test_read_LDM_file()




