"""Crystals module.

Data structures for crystal information.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import copy
from ctypes import ArgumentError
import numpy as np

#--- Model Imports ---
import crystal_calc
import numpy_utils
from numpy_utils import column, vector_length
import ubmatrixreader

#--- Traits Imports ---
from enthought.traits.api import HasTraits,Int,Float,Str,String,Property,Bool, List, Tuple, Array, Enum


# ===========================================================================================
class Crystal(HasTraits):
    """The Crystal class holds information about a crystal
    that a user is using/planning to use in an experiment.
    """
    name = Str("name")
    description = Str("description")

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
    #Same info, as a column-wise a,b,c matrix
    reciprocal_lattice = Array( shape=(3,3), dtype=np.float)

    #For loading the UB matrix
    ub_matrix_last_filename = Str

    #The hkl indices of interest. A list of (h,k,l) tuples.
    important_hkl = List

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
        #Pick the first long name
        self.point_group_name = get_point_group_names(long_name=True)[0]

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
            original_U = self.calculate_u_matrix(ub_matrix)

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


    #--------------------------------------------------------------------
    def calculate_reciprocal(self):
        """Calculate the reciprocal lattice of this crystal, from the direct
        lattice parameters."""
        (self.recip_a, self.recip_b, self.recip_c) = \
            crystal_calc.make_reciprocal_lattice(self.lattice_lengths, self.lattice_angles)
        #Also make the matrix
        self.reciprocal_lattice = numpy_utils.vectors_to_matrix(self.recip_a, self.recip_b, self.recip_c)

    #--------------------------------------------------------------------
    def get_u_matrix(self):
        """Return the previously-calculated U matrix."""
        return self.u_matrix
    

    #--------------------------------------------------------------------
    def calculate_u_matrix(self, ub_matrix):
        """Calculate the sample's U-matrix: the matrix describing the sample mounting orientation.
        So U = UB * (B)^-1

        Parameters:
            ub_matrix: ISAW-style UB matrix, after transposing and 2*pi.
                Coordinates will be transformed from IPNS convention to SNS convention

        """
        #The reciprocal_lattice lattice is the same as the B matrix
        B = self.reciprocal_lattice
        #Try to invert it
        try:
            invB = np.linalg.inv(B)
            U = np.dot(ub_matrix, invB)
            #Test that U must be orthonormal.
            U2 = np.dot(U, U.transpose())
            assert np.allclose(U2, np.eye(3), atol=1e-4), "The U matrix must be orthonormal. Instead, we got:\nU*U.transpose()=%s" % U2
            #Okay, now let's permute the rows for IPNS->SNS convention
            U_out = 1. * U
            U_out[2] = U[0] #x gets put in z
            U_out[1] = U[2] #z gets put in y
            U_out[0] = U[1] #y gets put in x
            U = U_out
            #Do another test
            U2 = np.dot(U, U.transpose())
            assert np.allclose(U2, np.eye(3), atol=1e-4), "The U matrix must be orthonormal. Instead, we got:\nU*U.transpose()=%s" % U2

            return U
        except np.linalg.LinAlgError:
            raise Error("Invalid reciprocal lattice found; B matrix could not be inverted.")

    #--------------------------------------------------------------------
    def get_B_matrix(self):
        """Returns the B matrix."""
        return  self.reciprocal_lattice


#================================================================================
#============================ GENERATORS ======================================
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

    Taken from Giacovazzo, Fundamentals of Crystallography, p.43
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
        """Contructor

        Parameters:
            name: string, short name of the point group, e.g. "4mm".
                Use a "-" as a prefix to indicate "bar", e.g. 1-bar = "-1"

            long_name: string, longer description of the point group
            args: one or more 3x3 generator matrices names, as strings, e.g. "2;1,0,0".
                use a "-" to indicate a negative generator, "-2;1,0,0"
            expected_order: order of the point group (aka number of matrices in the
                multiplication table). An error is raised if the calculation
                does not yield this expected number
            preferred_axis: string, 'h', 'k', or 'lkh', the rotation axis to prioritize
                when choosing which peaks are 'primary' ones
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

    Generator matrices taken from Giacovazzo, Fundamentals of Crystallography, p.46
    Generator choice from IUCR International Tables for Crystallography, Vol A, ch.8.3
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

    if False:
        #The following point groups do not have an inversion center,
        #   and therefore are not meaningful in diffraction experiments
        point_groups.append( PointGroup("1", "identity", 1, 'lkh', "1") )
        point_groups.append( PointGroup("2", "two-fold rotation along y axis", 2, 'k', "2;0,1,0"))
        point_groups.append( PointGroup("m", "mirror operation in the xy plane", 2, 'k', "-2;0,0,1"))
        point_groups.append( PointGroup("3", "three-fold rotation along z axis", 3, 'lkh', "H3;0,0,1"))
        point_groups.append( PointGroup("4", "four-fold rotation along z axis", 4, 'lkh', "4;0,0,1"))
        point_groups.append( PointGroup("6", "six-fold rotation along z axis", 6, 'lkh', "H6;0,0,1"))
        point_groups.append( PointGroup("-4", "four-fold rotoinversion along z axis", 4, 'lkh', "-4;0,0,1"))
        point_groups.append( PointGroup("-6", "six-fold rotoinversion along z axis", 6, 'lkh', "-H6;0,0,1"))
        point_groups.append( PointGroup("222", "", 4, 'lkh', "2;1,0,0", "2;0,1,0" ))
        point_groups.append( PointGroup("mm2", "", 4, 'lkh', "-2;1,0,0", "2;0,0,1" ))
        point_groups.append( PointGroup("32", "", 6, 'lkh', "H3;0,0,1", "2;1,1,0" )) #Changed 2nd generator from book
        point_groups.append( PointGroup("3m", "", 6, 'lhk', "H3;0,0,1", "-2;1,1,0" )) #Changed 2nd generator from book!
        point_groups.append( PointGroup("422", "", 8, 'lkh', "2;1,0,0", "2;1,-1,0" ))
        point_groups.append( PointGroup("4mm", "", 8, 'lkh', "-2;1,0,0", "-2;1,-1,0" ))
        point_groups.append( PointGroup("-62m", "", 12, 'lkh', "-2;1,0,0", "2;1,-1,0" ))
        point_groups.append( PointGroup("622", "", 12, 'lkh', "2;1,0,0", "2;1,-1,0" ))
        point_groups.append( PointGroup("6mm", "", 12, 'lkh', "-2;1,0,0", "-2;1,-1,0" ))
        point_groups.append( PointGroup("23", "", 12, 'lhk', "3;1,1,1", "2;0,0,1" ))
        point_groups.append( PointGroup("432", "", 24, 'lkh', "3;1,1,1", "2;1,1,0" ))
        point_groups.append( PointGroup("-43m", "", 24, 'lkh', "3;1,1,1", "-2;1,1,0" ))


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
#============================ UNIT TESTING ======================================
#================================================================================
import unittest
from numpy import pi

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
            assert hkl_wanted in found, "hkl %s was found in the list of results for %s" % (hkl_wanted, message)

    def test_point_groups(self):
        """Exhaustive and exhausting test each of the 11 Laue classes equivalent hkls"""
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


#---------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()



