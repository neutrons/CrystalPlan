"""Crystal calculations modules.

Holds various useful functions for crystallography, like lattice calculations, etc.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import numpy as np
from numpy import array, sin, cos, pi, sign
import weave

#--- Model Imports ---
import numpy_utils
from numpy_utils import column, rotation_matrix, vector_length, normalize_vector, vector, \
                    vectors_to_matrix, az_elev_direction, within


#========================================================================================================
def get_scattered_q_vector(hkl, rot_matrix, ub_matrix):
    """Find the scattered q-vector directions matching the given hkl indices.

        The fundamental equation is:
        normalized_direction = lambda . OMEGA . CHI . PHI . U . B . (hkl)
        normalized_direction = lambda . rot_matrix . ub_matrix . (hkl)

    Parameters:
        hkl: 3xN array containing the h,k,l reflections of interest. Columns (first dimension) are each (h,k,l).
        rot_matrix: The rotation matrix corresponding to the sample orientation (phi, chi, omega)
        ub_matrix: B converts hkl to q-vector, and U converts

    Returns:
        (q_vector) : 3xN array giving the scattered q-vector for each hkl (in each column).
    """
    matrix = np.dot(rot_matrix, ub_matrix)
    q_vector = np.dot(matrix, hkl)
    return q_vector


#========================================================================================================
def get_scattered_beam(hkl, rot_matrix, ub_matrix):
    """Find the scattered beam direction for the given hkl indices.

    Parameters:
        hkl: 3xN array containing the h,k,l reflections of interest. Columns (first dimension) are each (h,k,l).
        rot_matrix: The rotation matrix corresponding to the sample orientation (phi, chi, omega)
        ub_matrix: of the sample. B converts hkl to q-vector, and U orients.

    Returns:
        (beam) : non-normalized 3xN array giving the direction of the scattered beams in the instrument
            coordinates (+Z is beam direction, +Y is up).
            Length of vector = 2*pi / wavelength.

    Algorithm
    ---------
    First, find the q-vector direction:
        The fundamental equation is:
        q_vector = OMEGA . CHI . PHI . U . B . (hkl)
        q_vector = rot_matrix . ub_matrix . (hkl)
            (where the B matrix contains the 2*pi factor)
    
    Known: the q-vector you want to detect. (qx, qy, qz) components of the vector
    Known: ki, the incident beam wave vector, is in the direction +Z, length 2pi/wavelength
    Unknown: wavelength
    Unknown: kf, the scattered beam wave vector
    Known: kf - ki = q; or, kf = ki + q; This means that kf = qx in x; qy in y; and (qz+2pi/wl) in z.
    Known: norm(kf) = norm(ki) = 2*pi/wavelength

    With some algebra, we find that
        2pi/wl = - norm(q)^2 / (2*qz)
    So this goes into the kf_z = (qz+2pi/wl) 
    """
    # Calculate the scattered q-vector.
    matrix = np.dot(rot_matrix, ub_matrix)
    # This is the scattered Q in the L frame. 
    q_vector = np.dot(matrix, hkl)
    squared_norm_of_q = np.sum( q_vector**2, axis=0)
    # 2pi/wl = - norm(q)^2 / (2*qz)
    two_pi_over_wl = -0.5 * squared_norm_of_q / q_vector[2,:]
    #Calculate the scattered beam direction
    beam = q_vector.copy()
    #Add 2pi/wl to the Z component only
    beam[2,:] += two_pi_over_wl
    #if the qz is positive, then that scattering does not happen.
    #   set those to nan.
    beam[2, (q_vector[2,:]>0) ] = np.nan
    #And return it.
    return beam



#========================================================================================================
def get_sample_rotation_matrix_to_get_beam(beam_wanted, hkl, ub_matrix, starting_rot_matrix=None):
    """Find the sample rotation matrix that will result in a scattered beam
    going in the desired direction.

    Parameters:
        beam_wanted: vector containing a NORMALIZED vector of the DESIRED scattered beam direction.
        hkl: vector containing the h,k,l reflection of interest.
        ub_matrix: for the current sample and mounting. B converts hkl to q-vector, and U orients.
        starting_rot_matrix: optional: rotation matrix that is close to the result
            we are looking for. e.g. we want the beam to move a few degrees away from this
            starting sample orientation.

    Returns:
        rot_matrix: the 3x3 rotation matrix for the sample orientation that satisfies the
            desired beam direction; or None if it cannot be done.
        wavelength: wavelength at which it would be measured.

    Algorithm
    ---------
    - First, we find the _direction_ of q, ignoring the wavelength
        kf - ki = delta_q
        kf = two_pi_over_wl * (beam_wanted)
        ki = two_pi_over_wl (in z only)
        # so
        q/two_pi_over_wl = (kf - ki)/two_pi_over_wl = beam_x in x; beam_y in y; beam_z-1 in z
    """

    # a means two_pi_over_wl

    #First, we find the _direction_ of q, ignoring the wavelength
    q_over_a = beam_wanted.ravel()
    #Subtract 1 from Z to account for ki, incident beam
    q_over_a[2] -= 1

    #Find the q vector from HKL before sample orientation
    q0 = np.dot(ub_matrix, hkl)

    if not starting_rot_matrix is None:
        #Also rotate using the given matrix, to find a starting point
        q0 = np.dot(starting_rot_matrix, q0)

    #Flatten into vector
    q0 = q0.flatten()

    #Since the rotation does not change length, norm(q0) = norm(q_over_a),
    #   meaning that a = norm(q)/ norm(q_over_a)
    a = vector_length(q0) / vector_length(q_over_a)
    #And a's definition is such that:
    wavelength = 2*np.pi/a

    #So lets get q directly.
    q_rotated = q_over_a * a

    #Find the rotation matrix that satisfies q_over_a = R_over_a . q0
    
    #The cross product of q0 X q_over_a gives a rotation axis to use
    rotation_axis = np.cross(q0, q_over_a)
    #Normalize
    rotation_axis = rotation_axis / vector_length(rotation_axis)

    #Now we find the rotation angle about that axis that puts q0 on q_over_a
    angle = np.arccos( np.dot(q0, q_over_a) / (vector_length(q0)*vector_length(q_over_a)))

    #From http://www.euclideanspace.com/maths/algebra/matrix/orthogonal/rotation/index.htm
    # we make a rotation matrix
    (x,y,z) = rotation_axis

    #This is a normalized rotation matrix
    R = np.array([[1 + (1-cos(angle))*(x*x-1), -z*sin(angle)+(1-cos(angle))*x*y, y*sin(angle)+(1-cos(angle))*x*z],
        [z*sin(angle)+(1-cos(angle))*x*y, 1 + (1-cos(angle))*(y*y-1), -x*sin(angle)+(1-cos(angle))*y*z],
        [-y*sin(angle)+(1-cos(angle))*x*z,  x*sin(angle)+(1-cos(angle))*y*z,  1 + (1-cos(angle))*(z*z-1)]])

    if not starting_rot_matrix is None:
        #The final rotation we want is sstarting_rot_matrix 1st; R second.
        # So this is the resulting matrix
        R = np.dot(R, starting_rot_matrix)


#    print "q0", q0.ravel()
#    print "two pi over wl (a)", a
#    print "q_over_a", q_over_a.ravel()
#    print "rotation_axis", rotation_axis
#    print "angle", angle
#    print "get_sample_rotation_matrix_to_get_beam: angle was", np.rad2deg(angle)

    return (R, wavelength)




#========================================================================================================
def make_reciprocal_lattice(lattice_lengths, lattice_angles):
    """Generate a reciprocal lattice matrix.

    Parameters:
        lattice_lengths: tuple. the DIRECT lattice lengths a,b,c in Angstroms.
        lattice_angles: tuple. the DIRECT lattice lengths alpha, beta, gamma, in radians.
            alpha = angle between b and c; beta = angle between a and c; gamma = angle between b and c

    Returns:
        (a_star, b_star, c_star): the 3 reciprocal lattice vectors.
    """
    #First, get the direct space lattice vectors
    (a,b,c, V) = make_lattice_vectors(lattice_lengths, lattice_angles)

    #With cross-products, calculate the reciprocal vectors
    a_star = 2 * np.pi * np.cross(vector(b),vector(c)) / V
    b_star = 2 * np.pi * np.cross(vector(c),vector(a)) / V
    c_star = 2 * np.pi * np.cross(vector(a),vector(b)) / V

    return (a_star, b_star, c_star)


#========================================================================================================
def make_lattice_vectors(lattice_lengths, lattice_angles):
    """Generate lattice vectors from lengths and angles.

    Parameters:
        lattice_lengths: tuple. the lattice lengths a,b,c in Angstroms.
        lattice_angles: tuple. the lattice lengths alpha, beta, gamma, in radians.
            alpha = angle between b and c; beta = angle between a and c; gamma = angle between b and c

    Returns:
        a_vec, b_vec, c_vec: 3 column-wise vectors
            If invalid angles are given, c-vector may have a NaN.
        V: volume of the unit cell
    """
    #Get the direct lattice parameters
    (a,b,c) = lattice_lengths
    (alpha, beta, gamma) = lattice_angles

    #Make the G metric matrix (tensor)
#        G =np.array( [ [a*a, a*b*cos(gamma), a*c*cos(beta)],
#                       [a*b*cos(gamma), b*b, b*c*cos(alpha)],
#                       [a*c*cos(beta), b*c*cos(alpha), c*c] ] )
    # V = np.sqrt( np.linalg.det( G ) )

    #Unit cell volume (Fundamentals of Crystallography, Giacovazzo editor, p.64)
    V = a*b*c * np.sqrt(1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2*cos(alpha)*cos(beta)*cos(gamma))

    #V can be nan if bad parameters are passed in!

    #We take the "a" direction to be parallel with the x axis.
    a_vec = column( [a, 0, 0] )
    #We take the b direction to be roughly towards +y, in the XY plane
    b_vec = column( [b*cos(gamma),  b*sin(gamma), 0] )
    #Finally, the c direction points towards z positive, but is not otherwise constrained
    #  solution from Fundamentals of Crystallography, Giacovazzo editor, p.64, and 68
    cos_alpha_star = (cos(beta)*cos(gamma)-cos(alpha)) / (sin(beta)*sin(gamma))
    c_star = (a*b*sin(gamma)) / V
    c_vec = column( [c*cos(beta), -c*sin(beta)*cos_alpha_star, 1/c_star] )

    return (a_vec, b_vec, c_vec, V)



#========================================================================================================
def make_UB_matrix(lattice_lengths, lattice_angles, sample_phi, sample_chi, sample_omega,
                   U_matrix=None):
    """Make a UB matrix from the given parameters.

    Parameters:
        lattice_lengths: the DIRECT lattice lengths a,b,c in Angstroms.
        lattice_angles: tuple. the DIRECT lattice lengths alpha, beta, gamma, in radians.
            alpha = angle between b and c; beta = angle between a and c; gamma = angle between b and c

        sample_phi, sample_chi, sample_omega: These three angles (in radians) determine how
            the crystal is mounted. Here's how it works:
                0 - Instrument coordinates: +Y is vertical, +Z is the beam direction.
                1 - Start with the crystal mounted so that 'a' (real lattice vector)
                    is exactly aligned towards +X, and 'b' is pointing towards +Y in the XY plane,
                    and 'c' is pointing towards +Z
                2 - Do the phi rotation = around the Y axis (vertical).
                3 - Do the chi rotation = around the Z axis.
                4 - Do the omega rotation = around the Y axis again.
            For example, if 'a' is pointing up, use 0, +90, 0.
                if 'a' is pointing down along the beam direction, use -90, 0, 0 (I think)
                
        U_matrix : specify to use this U matrix instead of the angles
    """
    #Get the reciprocal vectors
    (a_star, b_star, c_star) = make_reciprocal_lattice(lattice_lengths, lattice_angles)

    #The B matrix is such that B.hkl = the q-vector (minus some rotations)
    #   so, B is a matrix with each column = to a_star, b_star, c_star
    #   B.hkl = a_star*h + b_star*k + c_star*l
    B = numpy_utils.vectors_to_matrix(a_star, b_star, c_star)

    #Now lets make a rotation matrix U to account for sample mounting.
    #   We used the same set of phi,chi,omega rotations as for sample orientations.
    U = numpy_utils.rotation_matrix(sample_phi, sample_chi, sample_omega)
    
    if not U_matrix is None:
        U = U_matrix

    #Return the UB matrix product. U*Bget_q_from_hkl
    return np.dot(U, B)


#========================================================================================================
def get_q_from_hkl(hkl, a_star, b_star, c_star):
    """Calculate a set of q-vector from a set of hkl indices.

    Parameters:
        hkl: a 3xN array of the hkl indices of the reflection.
        (a_star, b_star, c_star): The 3 reciprocal lattice vectors (3 element array each)

    Returns:
        q: a 3xN array of Q vectors, column-wise
    """
    #Make the B matrix
    B = numpy_utils.vectors_to_matrix(a_star, b_star, c_star)
    #Matrix multiply B.hkl to get all the Qs
    return np.dot(B, hkl)


#========================================================================================================
def get_hkl_from_q(q, reciprocal_lattice):
    """Calculate a set of hkl values from a set of q vectors.

    Parameters:
        q: a 3xN array of the hkl indices of the reflection.
        reciprocal_lattice: The 3 reciprocal lattice vectors as columns in a matrix (aka the B matrix)

    Returns:
        hkl: a 3xN array of FLOAT hkl values = they are not rounded or anything.
    """
    #Get the inverse the B matrix to do the reverse conversion
    invB = np.linalg.inv(reciprocal_lattice)
    #Matrix multiply invB.hkl to get all the HKLs
    return np.dot(invB, q)


#========================================================================================================
def getq_python(azimuth, elevation, wl_output, rot_matrix, wl_input=None):
    """Find the q-vector corresponding to the azimuth, elevation and wavelength of the detector pixel.
    Uses only python and numpy code.

    Paramters:
        azimuth, elevation: azimuth, elevation angle of pixel(s). Can be an array, should be only 1-D though.
            Shapes of az and elev need to match.
        wl_output: output (scattered) wavelength considered; can be scalar or array matching az and elev.
        rot_matrix: The rotation matrix to apply to the q in the lab frame (should be the inverse of the goniometer rotation).
        wl_input: input (incident) wavelength; if not specified, defaults to wl_output

    Returns:
        q: q-vector, either a single column or a 3xn array depending on input size.
    """
    #The Ewald sphere has 1/wl radius
    inelastic = True
    if wl_input is None:
        inelastic = False
        wl_input = wl_output

    #The scattered beam emanates from the centre of this spher.
    #Find the intersection of the scattered beam and the sphere, in XYZ
    beam = column(az_elev_direction(azimuth, elevation)) / wl_output

    #And here is the incident beam direction: Along the z-axis, positive
    incident = np.array([0, 0, 1.0]).reshape(3,1) / wl_input

    #The wave vector difference between the two is the q vector
    q = 2*pi * (beam - incident)

    #Now we switch to the coordinate system of the crystal.
    #The scattered beam direction (the detector location) is rotated relative to the crystal
    #   because the sample is rotated.
    #So is the incident beam direction.
    #Therefore, the q-vector measured is simply rotated by the supplied rotation matrix (which has reversed angles)

    if inelastic:
        q_unrotated = q
        q = np.dot(rot_matrix, q_unrotated)
        return (q, q_unrotated)
    else:
        q = np.dot(rot_matrix, q)
        return q


#========================================================================================================
#Code for the getq command, in C

#Header for elastic scattering (same wavelengths)
getq_code_header = """py::tuple getq(double wl_output, double az, double elevation, double pi, double* rot_matrix)
{
    py::tuple return_val(3);
    double wl_input = wl_output;
"""


#Core code, common for both elastic and inelastic
getq_code = """
    double r2, x, y, z;
    
    // The scattered beam emanates from the centre of this spher.
    // Find the intersection of the scattered beam and the sphere, in XYZ
    // We start with an Ewald sphere of radius 1/wavelength

    //Assuming azimuth of zero points to z positive = same direction as incident radiation.
    r2 = cos(elevation) / wl_output;
    z=cos(az) * r2;
    x=sin(az) * r2;

    // Assuming elevation angle is 0 when horizontal, positive to y positive:
    y=sin(elevation) / wl_output;

    //And here is the incident beam direction: Along the z-axis
    double incident_z = 1.0 / wl_input;

    //The vector difference between the two is the q vector
    float qx, qy, qz;
    qx = 2 * pi * x;
    qy = 2 * pi * y;
    qz = 2 * pi * (z - incident_z);

    //#Now we switch to the coordinate system of the crystal.
    //#The scattered beam direction (the detector location) is rotated relative to the crystal because the sample is rotated.
    //#So is the incident beam direction.
    //#Therefore, the q-vector measured is simply rotated

    //Here we perform the rotation by doing a matrix multiplication.
    py::tuple q(3);
    q[0]=qx * rot_matrix[0+0] + qy * rot_matrix[0+1] + qz * rot_matrix[0+2];
    q[1]=qx * rot_matrix[3+0] + qy * rot_matrix[3+1] + qz * rot_matrix[3+2];
    q[2]=qx * rot_matrix[6+0] + qy * rot_matrix[6+1] + qz * rot_matrix[6+2];
    return_val = q;
"""
getq_code_footer = "return return_val; }"




#Header/footer for inelastic scattering
getq_inelastic_code_header = """py::tuple getq_inelastic(double wl_input, double wl_output, double az, double elevation, double pi, double* rot_matrix)
{
    py::tuple return_val(6);
"""

getq_inelastic_code = """
    double r2, x, y, z;
    //Assuming azimuth of zero points to z positive = same direction as incident radiation.
    r2 = cos(elevation) / wl_output;
    z=cos(az) * r2;
    x=sin(az) * r2;

    // Assuming elevation angle is 0 when horizontal, positive to y positive:
    y=sin(elevation) / wl_output;

    //And here is the incident beam direction: Along the z-axis
    double incident_z = 1.0 / wl_input;

    //The vector difference between the two is the q vector
    float qx, qy, qz;
    qx = 2 * pi * x;
    qy = 2 * pi * y;
    qz = 2 * pi * (z - incident_z);

    //Here we perform the rotation by doing a matrix multiplication.
    py::tuple q(6);
    q[0]=qx * rot_matrix[0+0] + qy * rot_matrix[0+1] + qz * rot_matrix[0+2];
    q[1]=qx * rot_matrix[3+0] + qy * rot_matrix[3+1] + qz * rot_matrix[3+2];
    q[2]=qx * rot_matrix[6+0] + qy * rot_matrix[6+1] + qz * rot_matrix[6+2];
    //Unrotated
    q[3]=qx;
    q[4]=qy;
    q[5]=qz;
    return_val = q;
    """
    
getq_inelastic_code_footer = """
    return return_val;
}
"""

#========================================================================================================
def getq(az, elevation, wl_output, q_rot_matrix, wl_input=None):
    """Find the q-vector corresponding to the azimuth, elevation and wavelength of the detector pixel.
    Optimized for speed by using inline C code. Provides a ~45% time reduction for coverage calculation.

    Paramters:
        az, elevation: azimuth, elevation angle of pixel. Scalar only.
        wl_output: output (scattered) wavelength considered; can be scalar or array matching az and elev.
        q_rot_matrix: The q-rotation matrix; how the q-vector has to be rotated.
            This corresponds to the opposite angles of the sample orientation,
            so (-phi, -chi, -omega).
        wl_input: input (incident) wavelength; if not specified, defaults to wl_output

    Returns:
        q: q-vector, a single column
    """

    support = """
    #include <math.h>
    """
    #Ensure the right data types!
    az = float(az)
    elevation = float(elevation)
    wl_output = float(wl_output)
    rot_matrix = q_rot_matrix
    if wl_input is None:
        # -- elastic ---
        wl_input = wl_output
        q = weave.inline(getq_code, ['wl_input', 'wl_output', 'elevation', 'az', 'pi', 'rot_matrix'],compiler='gcc', support_code = support,libraries = ['m'])
        q = column([q[0],q[1],q[2]])
        return q
    else:
        #--- inelastic ---
        (q_both) =  weave.inline(getq_inelastic_code, ['wl_input', 'wl_output', 'elevation', 'az', 'pi', 'rot_matrix'],compiler='gcc', support_code = support,libraries = ['m'])
        q = np.array(q_both[0:3]).reshape(3,1)
        q_unrot = np.array(q_both[3:]).reshape(3,1)
        return (q, q_unrot)



# ================= NIGGLI MATRIX CODE =======================

#Tolerance for equalities and inequalities.
NIGGLI_TOLERANCE = 0.01

def niggli_g1(A, B, C, D, E, F):
    """Inner part of Niggli algorithm.
    This function is called recursively, as per the algorithm in
        Acta. Cryst. (1976) A32, 297, by Krivy and Gruber.
    Returns A thru F."""

    #Define a tolerance for inequalities and equalities
    tolerance=NIGGLI_TOLERANCE
    #G1
    if A>B+tolerance or (within(A,B) and abs(D)>abs(E)):
        A,B = B,A   #swap A and B
        D,E = E,D   #swap D and E too
    #G2
    if B>C+tolerance or (within(B,C) and abs(E)>abs(F)):
        B,C = C,B
        E,F = F,E
        (A, B, C, D, E, F) = niggli_g1(A, B, C, D, E, F) #Goto G1
    #G3
    if D*E*F > 0:
        D=abs(D)
        E=abs(E)
        F=abs(F)
    #G4
    if D*E*F <= 0:
        D=-abs(D)
        E=-abs(E)
        F=-abs(F)
    #G5
    if abs(D)>B+tolerance or (within(D,B) and 2*E<F) or (within(D,-B) and F<0):
        C=B+C-D*sign(D)
        E=E-F*sign(D)
        D=D-2*B*sign(D)
        (A, B, C, D, E, F) = niggli_g1(A, B, C, D, E, F) #Goto G1
    #G6
    if abs(E)>A+tolerance or (within(E,A) and (2*D)<F) or (within(E,-A) and F<0):
        C=A+C-E*sign(E)
        D=D-F*sign(E)
        E=E-2*A*sign(E)
        (A, B, C, D, E, F) = niggli_g1(A, B, C, D, E, F) #Goto G1
    #G7
    if abs(F)>A+tolerance or (within(F,A) and (2*D<E)) or (within(F,-A) and E<0):
        B=A+B-F*sign(F)
        D=D-E*sign(F)
        F=F-2*A*sign(F)
        (A, B, C, D, E, F) = niggli_g1(A, B, C, D, E, F) #Goto G1
    #G8
    if (D+E+F+A+B)<0 or (abs(D+E+F+A+B)<tolerance and (2*(A+E)+F)>0):
        C=A+B+C+D+E+F
        D=2*B+D+F
        E=2*A+E+F
        (A, B, C, D, E, F) = niggli_g1(A, B, C, D, E, F) #Goto G1

    #Go on to the exit
    return (A, B, C, D, E, F)


def make_niggli_matrix(a_in,b_in,c_in,alpha,beta,gamma):
    """Calculate the Niggli matrix from the lattice parameters supplied.
    Parameters:
        (a_in,b_in,c_in): The lattice parameters are a,b,c
        (alpha, beta, gamma): lattice angles in degrees

    Based on the algorithm in:
        Acta. Cryst. (1976) A32, 297, by Krivy and Gruber.
    The variables A thru F fill the Niggli Matrix thus:
        [[A, B, C], [D/2, E/2, F/2]]"""

    #G0: Start by reading in the values
    A = a_in*a_in
    B = b_in*b_in
    C = c_in*c_in
    D = 2*b_in*c_in*cos(np.deg2rad(alpha))  #Or vector dot product b.c
    E = 2*c_in*a_in*cos(np.deg2rad(beta))   #c.a
    F = 2*a_in*b_in*cos(np.deg2rad(gamma))  #a.b
    #Go to G1
    (A, B, C, D, E, F) = niggli_g1(A, B, C, D, E, F)
    #Done!
    return array([[A, B, C], [D/2, E/2, F/2]])











#================================================================================
#============================ UNIT TESTING ======================================
#================================================================================
import unittest


#==================================================================
class TestCrystalCalc(unittest.TestCase):
    """Unit test for the crystal_calc module."""

    def test_getq(self):
        """cystal_calc module: getq()."""
        #Zero-level test
        q1 = getq(0, 0, 1.0, rotation_matrix(0,0,0))
        assert not np.any(q1), "Non-scattered beam gives a q of 0,0,0"

        q1 = getq(-np.pi, 0.0, 1.0, rotation_matrix(0,0,0)).flatten()
        answer = np.array([0,0,-2.])*2*np.pi
        assert np.allclose(q1, answer), "Back-scattered beam gives a q of %s; result was %s" % (answer, q1)

        q1 = getq(-np.pi, 0.0, 2.0, rotation_matrix(0,0,0)).flatten()
        answer = np.array([0,0,-1.])*2*np.pi
        assert np.allclose(q1, answer), "Back-scattered beam at wl = 2 Angstrom gives a q of %s; result was %s" % (answer, q1)

        q1 = getq(-np.pi, 0, 3, rotation_matrix(0,0,0)).flatten()
        answer = np.array([0,0, -2.0/3.0])*2*np.pi
        assert np.allclose(q1, answer), "Getq handles integer inputs."

        q1 = getq(0, np.pi/2, 1.0, rotation_matrix(0,0,0)).flatten()
        answer = np.array([0,+1.,-1.])*2*np.pi
        assert np.allclose(q1, answer), "Vertically-scattered beam gives a q of %s; result was %s" % (answer, q1)

    def test_getq_with_rotation(self):
        "cystal_calc module: getq() test with sample rotation. The q-rotation matrix always uses OPPOSITE angles"

        #Rotate sample +pi/2 in phi.
        q_rot_m = rotation_matrix(-np.pi/2,0,0)
        q1 = getq(-np.pi, 0.0, 1.0, q_rot_m).flatten()
        answer = np.array([+2.,0.,0.])*2*np.pi
        assert np.allclose(q1, answer), "Back-scattered beam, with phi rotation, gives a q of %s; result was %s" % (answer, q1)

        q1 = getq(0, pi/2, 1.0, q_rot_m).flatten()
        answer = np.array([+1.,+1.,0.])*2*np.pi
        assert np.allclose(q1, answer), "Vertically-scattered beam, with phi rotation, gives a q of %s; result was %s" % (answer, q1)

        #Rotate sample +pi/2 in chi. No effect on back-scattered
        q_rot_m = rotation_matrix(0,-np.pi/2,0)
        q1 = getq(-np.pi, 0.0, 1.0, q_rot_m).flatten()
        answer = np.array([0.,0.,-2.])*2*np.pi
        assert np.allclose(q1, answer), "Back-scattered beam, with chi rotation, gives a q of %s; result was %s" % (answer, q1)
        q_rot_m = rotation_matrix(0,0.2345,0)
        q1 = getq(-np.pi, 0.0, 1.0, q_rot_m).flatten()
        answer = np.array([0.,0.,-2.])*2*np.pi
        assert np.allclose(q1, answer), "Back-scattered beam, with any chi rotation, gives a q of %s; result was %s" % (answer, q1)


#        q1 = getq(0, 1.0, 0.0, rotation_matrix(0,0,0))
#        assert np.allclose(q1, (1.0, "Vertically scattered beam gives a q of 0,0,0"


    def test_getq_c_and_python(self):
        """cystal_calc module: getq() in C and python."""
        #Angles for testing
        az = 1.23
        elev = 0.32
        wl = 2.32
        rot_matrix = rotation_matrix(0.31, -0.23, 0.43)
        q1 = getq_python(az, elev, wl, rot_matrix)
        q2 = getq(az, elev, wl, rot_matrix)
        assert np.allclose(q1, q2) , "Python and C getq match within float error."
        #assert np.all(q1 == q2) , "Python and C getq match exactly."

        #Now test for an array
        az = np.linspace(0, pi, 50)
        elev = az / 2
        rot_matrix = rotation_matrix(0.31, -0.23, 0.43)
        several_q = getq_python(az, elev, wl, rot_matrix)
        assert several_q.shape == (3, 50), "Correct shape"
        wl = az + 1
        several_q = getq_python(az, elev, wl, rot_matrix)
        assert several_q.shape == (3, 50), "Correct shape with wl as array"


    def test_getq_inelastic(self):
        #Angles for testing
        az = 1.23
        elev = 0.32
        wl = 2.32
        wl_input = 1.02
        rot_matrix = rotation_matrix(0.31, -0.23, 0.43)
        q1 = getq_python(az, elev, wl, rot_matrix, wl_input=wl_input)
        q2 = getq(az, elev, wl, rot_matrix, wl_input=wl_input)
        print "python:", q1
        print "c:", q2
        assert np.allclose(q1[0].flatten(), np.array(q2[0]).flatten()) , "Python and C getq (inelastic) match within float error. %s (python); %s (C)" % (q1, q2)
        assert np.allclose(q1[1].flatten(), np.array(q2[1]).flatten()) , "Python and C getq (inelastic) match within float error. %s (python); %s (C)" % (q1, q2)
        #assert np.all(q1 == q2) , "Python and C getq match exactly."


    def test_scattered_beam(self):
        """cystal_calc module: get_scattered_q_vector() tests."""
        #Easy UB matrix test
        UB = np.identity(3)
        rot_matrix = rotation_matrix(0, 0, 0)
        #Single vector
        hkl = column([1,2,3])
        scattered_q = get_scattered_q_vector(hkl, rot_matrix, UB)
        assert scattered_q.shape==(3,1), "Returns column-wise vector."
        assert np.allclose(scattered_q, column([1.,2.,3.])), "Identity operation."
        #Several vectors
        hkl = column([1,2,3]) + np.arange(0, 10)
        scattered_q = get_scattered_q_vector(hkl, rot_matrix, UB)
        assert scattered_q.shape==(3,10), "Returns column-wise array."


    def test_getq_vs_get_scattered_beam(self):
        "test_getq_vs_get_scattered_beam: getq and get_scattered_beam should give compatible results."
        #Test depends on test_make_UB_matrix() and on test_get_q_from_hkl()
        
        #Make up a simple lattice - cubic
        lat = ( 1.0,  1.0,  1.0)
        angles = tuple( np.deg2rad( [90, 90, 90]))
        UB = make_UB_matrix(lat, angles, 0, 0, 0)
        #No sample rotation
        rot_matrix = np.identity(3)

        #Let's try a non-scattered beam. q=0,0,0
        hkl = column([0.,0.,0.])
        scattered_q = get_scattered_q_vector(hkl, rot_matrix, UB)
        assert np.allclose(scattered_q, [0,0,0]), "Non-scattered q-vector matches hkl 0,0,0"
        beam = get_scattered_beam(hkl, rot_matrix, UB)
        assert np.any(np.isnan(beam)), "Non-scattered beam direction is nan: %s" % beam.flatten()

        #Let's try a back-scattered beam.
        # So its q = 2 * (2*pi* 1/incident_wavelength) in the negative Z
        wl = 1.0 #1 Angstrom wl
        q1 = getq( -pi, 0, wl, rot_matrix)
        #Since the lattice is also 1 Angstrom, the reciprocal lattice is 2pi, so an l = -2
        hkl = column([0.,0.,-2.])
        scattered_q = get_scattered_q_vector(hkl, rot_matrix, UB)
        assert np.allclose(scattered_q, q1), "Back-scattered q-vector matches hkl %s" % hkl.flatten()
        beam = get_scattered_beam(hkl, rot_matrix, UB)
        answer = column([0,0,-1.0]) * (2*pi/wl)
        assert np.allclose(beam, answer), "Back-scattered beam direction is correct: %s" % answer.flatten()

        #Let's try a back-scattered beam at twice the wavelength
        wl = 2.0
        q1 = getq( -pi, 0, wl, rot_matrix)
        #Now a closer l should see it
        hkl = column([0.,0.,-1.])
        scattered_q = get_scattered_q_vector(hkl, rot_matrix, UB)
        assert np.allclose(scattered_q, q1), "Back-scattered q-vector matches hkl %s" % hkl.flatten()
        beam = get_scattered_beam(hkl, rot_matrix, UB)
        answer = column([0,0,-1.0]) * (2*pi/wl)
        assert np.allclose(beam, answer), "Back-scattered beam direction is correct: %s" % answer.flatten()

        #Opposite hkl
        hkl = column([0.,0.,+1.])
        beam = get_scattered_beam(hkl, rot_matrix, UB)
        assert np.any(np.isnan(beam)), "No beam for an hkl of (0,0,1)"

        #Vertical q?
        wl = 1.0
        q1 = getq( 0, np.pi/2, wl, rot_matrix)
        hkl = column([0.,+1.,-1.])
        scattered_q = get_scattered_q_vector(hkl, rot_matrix, UB)
        assert np.allclose(scattered_q, q1), "Vertically-scattered q-vector matches hkl %s" % hkl.flatten()
        beam = get_scattered_beam(hkl, rot_matrix, UB)
        answer = column([0,+1.0,0.0]) * (2*pi/wl)
        assert np.allclose(beam, answer), "Vertically-scattered beam direction is correct: %s" % answer.flatten()

        #Down q
        q1 = getq( 0, -np.pi/2, 1.0, rot_matrix)
        hkl = column([0.,-1.,-1.])
        scattered_q = get_scattered_q_vector(hkl, rot_matrix, UB)
        assert np.allclose(scattered_q, q1), "Vertically-down-scattered q-vector matches hkl %s" % hkl.flatten()
        beam = get_scattered_beam(hkl, rot_matrix, UB)
        answer = column([0,-1.0,0.0]) * (2*pi/wl)
        assert np.allclose(beam, answer), "Vertically-down-scattered beam direction is correct: %s" % answer.flatten()


    def test_get_q_from_hkl(self):
        lat = ( 5.0,  10.0,  20.0)
        lat_inverse = (0.2, 0.1, 0.05)
        angles = tuple( np.deg2rad( [90, 90, 90]))
        (a,b,c) = make_reciprocal_lattice(lat, angles)
        q = get_q_from_hkl(column([1., 0, 0]), a, b, c)
        assert np.allclose(vector(q), a), "Simple get_q_from_hkl a."
        q = get_q_from_hkl(column([1., 0, 0]), a, b, c)
        assert np.allclose(vector(q), vector([2*pi/5, 0, 0])), "Simple get_q_from_hkl a, comparing to value."
        q = get_q_from_hkl(column([0, 1., 0]), a, b, c)
        assert np.allclose(vector(q), b), "Simple get_q_from_hkl b."
        q = get_q_from_hkl(column([0, 0, 1.]), a, b, c)
        assert np.allclose(vector(q), c), "Simple get_q_from_hkl c."
        q = get_q_from_hkl(column([1, 2, 3.]), a, b, c)
        assert np.allclose(vector(q), a+2*b+3*c), "get_q_from_hkl (1,2,3)."

        hkl = np.ones( (3, 12) )
        q = get_q_from_hkl(hkl, a, b, c)
        assert q.shape == (3, 12), "get_q_from_hkl correct shape."

    def test_get_hkl_from_q(self):
        lat = ( 5.0,  10.0,  20.0)
        lat_inverse = (0.2, 0.1, 0.05)
        angles = tuple( np.deg2rad( [90, 90, 90]))
        (a,b,c) = make_reciprocal_lattice(lat, angles)
        rec = vectors_to_matrix(a,b,c)
        hkl = get_hkl_from_q(column([0.2, 0, 0])*2*pi, rec)
        assert np.allclose(vector(hkl), vector([1,0,0])), "get_hkl_from_q (1,0,0)."
        #Sanity check
        for i in xrange(10):
            val = column(np.random.rand(3))
            q = get_q_from_hkl(val, a, b, c)
            hkl = get_hkl_from_q(q, rec)
            assert np.allclose(vector(hkl), vector(val)), "get_hkl_from_q vs get_q_from_hkl for random values."


    def check_matrix_lengths(self, M, lat):
        """Check if the lengths of the columns of M match the values in tuple lat"""
        lengths = [(vector_length(M[:,x])) for x in xrange(3)]
        #print "lengths", lengths
        return np.allclose(lengths, lat)

    def test_make_lattice_vectors(self):
        "cystal_calc module: make_lattice_vectors()"
        #Make up a lattice - orthorombic
        lat = ( 5.0,  10.0,  20.0)
        angles = tuple( np.deg2rad( [90, 90, 90]))
        #Make the expected matrix
        M = np.identity(3)
        for x in xrange(3): M[x,x] = lat[x]
        (a,b,c,V) = make_lattice_vectors(lat, angles)
        res = (vectors_to_matrix(a,b,c))
        assert np.allclose(M, res), "Simple orthorombic lattice"
        assert self.check_matrix_lengths(res, lat), "orthorombic: vectors are correct length."

        #Monoclinic example
        angles = tuple( np.deg2rad( [45, 90, 90]))
        M[:,2] = np.sqrt(2)/2 * vector( 0, 20, 20 ) #Value by inspection
        (a,b,c,V) = make_lattice_vectors(lat, angles)
        res = (vectors_to_matrix(a,b,c))
        assert np.allclose(M, res), "Monoclinic lattice"
        assert self.check_matrix_lengths(res, lat), "Monoclinic: vectors are correct length."

        #Rhombohedral example
        angles = tuple( np.deg2rad( [45, 45, 45]))
        M = np.identity(3)
        for x in xrange(3): M[x,x] = lat[x]
        M[:,1] = np.sqrt(2)/2 * vector( 10, 10, 0 ) #Value by inspection
        M[:,2] = vector( 14.14213562, 5.85786438, 12.87188506 ) #Value came from program
        (a,b,c,V) = make_lattice_vectors(lat, angles)
        res = (vectors_to_matrix(a,b,c))
        assert np.allclose(M, res), "Rhombohedral lattice"
        assert self.check_matrix_lengths(res, lat), "Rhombohedral: vectors are correct length."

        #Triclinic example
        angles = tuple( np.deg2rad( [30, 45, 60]))
        M = np.identity(3)
        for x in xrange(3): M[x,x] = lat[x]
        M[:,1] = vector( 5, 8.66025404, 0 ) #Value came from program
        M[:,2] = vector( 14.14213562, 11.83503419, 7.74157385 ) #Value came from program
        (a,b,c,V) = make_lattice_vectors(lat, angles)
        res = (vectors_to_matrix(a,b,c))
        assert np.allclose(M, res), "Triclinic lattice"
        assert self.check_matrix_lengths(res, lat), "Triclinic: vectors are correct length."


    def test_make_reciprocal_lattice(self):
        "cystal_calc module: make_reciprocal_lattice()"
        #Make up a lattice - orthorombic
        lat = ( 5.0,  10.0,  20.0)
        lat_inverse = 2 * pi * np.array((0.2, 0.1, 0.05))
        angles = tuple( np.deg2rad( [90, 90, 90]))
        (a,b,c) = make_reciprocal_lattice(lat, angles)
        res = (vectors_to_matrix(a,b,c))
        M = np.array([ [0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.05] ]) * 2 * pi
        assert np.allclose(M, res), "orthorombic reciprocal lattice"
        assert self.check_matrix_lengths(res, lat_inverse), "orthorombic: reciprocal lattice matrix is correct."
        #Triclinic example
        angles = tuple( np.deg2rad( [30, 45, 60]))
        V = 335.219981051 #From calculation of the program
        lat_inverse = 2*pi*np.array( (200*sin( angles[0] )/V, 100*sin( angles[1] )/V, 50*sin( angles[2] )/V)) #From formulae
        (a,b,c) = make_reciprocal_lattice(lat, angles)
        res = (vectors_to_matrix(a,b,c))
        assert self.check_matrix_lengths(res, lat_inverse), "Triclinic: reciprocal vectors are correct length."


    def test_make_UB_matrix(self):
        "cystal_calc module: make_UB_matrix()"
        #Make up a lattice - orthorombic
        lat = ( 5.0,  10.0,  20.0)
        angles = tuple( np.deg2rad( [90, 90, 90]))
        UB = make_UB_matrix(lat, angles, 0, 0, 0)
        M = np.array([[0.2,0,0], [0,0.1,0], [0,0,0.05]]) * 2 * pi
        assert np.allclose(M, UB), "UB matrix for orthorombic lattice, no rotation"
        UB = make_UB_matrix(lat, angles, np.deg2rad(90), np.deg2rad(0), np.deg2rad(0))
        M = np.array([[0,0,0.05], [0,0.1,0], [-0.2,0,0]]) * 2 * pi
        assert np.allclose(M, UB), "UB matrix for orthorombic lattice, phi=+90"
        UB = make_UB_matrix(lat, angles, np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))
        M = np.array([[0,-0.1,0], [0.2,0,0], [0,0,0.05]]) * 2 * pi
        assert np.allclose(M, UB), "UB matrix for orthorombic lattice, chi=+90"

    def test_get_sample_rotation_matrix_to_get_beam(self):
        beam_wanted = column([0,1,0])
        ub_matrix = np.identity(3)
        hkl = column([0,1,1])
        (R, wl) = get_sample_rotation_matrix_to_get_beam(beam_wanted, hkl, ub_matrix)
        assert np.allclose(R, np.array([[1,0,0],[0,0,1],[0,-1,0]])), "get_sample_rotation_matrix, case 1"
#        #Lets give it a starting matrix
#        start_R = rotation_matrix(0,np.pi/4, 0)
#        (R, wl) = get_sample_rotation_matrix_to_get_beam(beam_wanted, hkl, ub_matrix, start_R)
#        assert np.allclose(R, np.array([[1,0,0],[0,0,1],[0,-1,0]])), "get_sample_rotation_matrix, with a starting matrix. We got %s" % R
        

#---------------------------------------------------------------------
if __name__ == "__main__":
    rot_matrix = numpy_utils.rotation_matrix(np.rad2deg(45), np.rad2deg(90), 0)
    print rot_matrix

    start = column([-1,0,0])
    print start
    
    rot = np.dot(rot_matrix, start)
    print rot
    #unittest.main()
