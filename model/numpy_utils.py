""" Numpy utils module.

Provides useful functions that extend the numpy module.
In particular, there is code to handle column vectors.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import numpy as np
from numpy import array, sin, cos, pi, dot



#==================================================================================
#======================== GENERAL ===============================================
#==================================================================================
def within(a, b, tolerance=1e-5):
    """Return True if a==b within the relative tolerance specified."""
    if a==0 and b==0:
        return True
    elif a!=0 and b==0:
        #Avoid divide by zero
        return False
    else:
        return abs((a/b)-1) < tolerance

#==================================================================================
#======================== ROTATIONS ===============================================
#==================================================================================

#===============================================================================================

def rotation_angle_axis(x=0, y=0, z=1, theta=0):
    
    ux, uy, uz = [x,y,z]/np.linalg.norm([x,y,z])

    c = cos(theta)
    s = sin(theta)
    
    M = np.array([[c+ux**2*(1-c), ux*uy*(1-c)-uz*s, uz*ux*(1-c)+uy*s], 
                  [ux*uy*(1-c)+uz*s, c+uy**2*(1-c), uy*uz*(1-c)-ux*s], 
                  [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz**2*(1-c)]])
    
    return M

def rotation_matrix(phi=0, chi=0, omega=0):
    """Generate a rotation matrix M for 3 rotation angles:
       Uses convention of IPNS and ISAW for angles.
       PHI = first rotation, around the y axis
       CHI = second rotation, around the z axis
       OMEGA = third rotation, around the y axis again.

       Angles in radians.
       Use rotated_vector = matrix * initial_vector"""

    #s and c are temp. variables for sin(x) and cos(x)
    c = cos(phi)
    s = sin(phi)
    M_phi = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    c = cos(chi)
    s = sin(chi)
    M_chi = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    c = cos(omega)
    s = sin(omega)
    M_omega = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    #rotated =  M_omega * (M_chi * (M_phi * vector));
    M = np.dot(M_omega, np.dot(M_chi, M_phi))

    return M;

#===============================================================================================
def kappa_rotation_matrix(phi=0, alpha=0, kappa=0, omega=0):
    """Generate a rotation matrix M for 3 rotation angles:
       Uses convention of IPNS and ISAW for angles.
       PHI = first rotation, around the y axis
       ALPHA = Constant for mini-kappa
       KAPPA = second rotation, around the z axis
       OMEGA = third rotation, around the y axis again.

       Angles in radians.
       Use rotated_vector = matrix * initial_vector"""

    #s and c are temp. variables for sin(x) and cos(x)
    c = cos(-phi)
    s = sin(-phi)
    M_phi = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # kappa for mini-kappa goniometer
    c = cos(kappa)
    s = sin(kappa)
    
    # See The Geometry of X-Ray Diffraction p. 180 for matrix
    M_kappa = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    M_alpha = rotation_angle_axis(x=1, y=0, z=1, theta=alpha)
    M_kappa = np.dot(M_alpha.T, np.dot(M_kappa, M_alpha))

    c = cos(omega)
    s = sin(omega)
    M_omega = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    #rotated =  M_omega * (M_kappa * (M_phi * vector));
    M = np.dot(M_omega, np.dot(M_kappa, M_phi))

    return M;


#===============================================================================================
def angles_from_rotation_matrix(rot_matrix):
    """Find the 3 rotation angles that give the provided rotation matrix.

    Return:
       PHI = first rotation, around the y axis
       CHI = second rotation, around the z axis
       OMEGA = third rotation, around the y axis again.

       Angles in radians.
       Use rotated_vector = matrix * initial_vector

    This is taken from the Goiniometer.getEulerAngles() function that is
    in the package gov.ornl.sns.translation.geometry.calc.jython
    """
    R = rot_matrix

    #Let's make 3 vectors describing XYZ after rotations
    u = np.dot(R, column([1,0,0])).ravel()
    v = np.dot(R, column([0,1,0])).ravel()
    n = np.dot(R, column([0,0,1])).ravel()

    #is v.y vertical?
    if np.allclose(v[1], +1.0, rtol=1e-8, atol=1e-10):
        #Chi rotation is 0, so we just have a rotation about y
        chi = 0
        phi = np.arctan2(n[0], n[2])
        omega = 0
    elif np.allclose(v[1], -1.0, rtol=1e-8, atol=1e-10):
        #Chi rotation is 180 degrees
        chi = np.pi
        phi = -np.arctan2(n[0], n[2])
        if phi==-np.pi: phi=np.pi
        omega = 0
    else:
        #General case
        phi = np.arctan2(n[1], u[1])    #atan2(n.y, u.y)
        chi = np.arccos(v[1])           #acos(v.y)
        omega = np.arctan2(v[2], -1.*v[0])      #atan2(v.z, -v.x)
    
    return (phi, chi, omega)

#===============================================================================================
def opposite_rotation_matrix(phi=0, chi=0, omega=0):
    """Generate the opposite rotation matrix that rotation_matrix() gives.
       Uses convention of IPNS and ISAW for angles.

       #1. Rotate by -OMEGA, around the y axis
       #2. Rotate by -CHI, around the z axis
       #3. Rotate by -PHI, around the y axis

       Angles in radians.
       Use rotated_vector = matrix * initial_vector"""

    #s and c are temp. variables for sin(x) and cos(x)
    c = cos(-phi)
    s = sin(-phi)
    M_phi = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    c = cos(-chi)
    s = sin(-chi)
    M_chi = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    c = cos(-omega)
    s = sin(-omega)
    M_omega = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    #Multiply matrices in the reversed order
    M = np.dot(M_phi, np.dot(M_chi, M_omega))

    return M;


#===============================================================================================
def kappa_opposite_rotation_matrix(phi=0, alpha=0, kappa=0, omega=0):
    """Generate the opposite rotation matrix that rotation_matrix() gives.
       Uses convention of IPNS and ISAW for angles.

       #1. Rotate by -OMEGA, around the y axis
       #2. Rotate by -KAPPA around the z axis
       #3. Rotate by -PHI, around the y axis

       Angles in radians.
       Use rotated_vector = matrix * initial_vector"""

    #s and c are temp. variables for sin(x) and cos(x)
    c = cos(phi)
    s = sin(phi)
    M_phi = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # kappa for mini-kappa goniometer
    c = cos(-kappa)
    s = sin(-kappa)

    # See The Geometry of X-Ray Diffraction p. 180 for matrix
    M_kappa = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    M_alpha = rotation_angle_axis(x=1, y=0, z=1, theta=alpha)
    M_kappa = np.dot(M_alpha.T, np.dot(M_kappa, M_alpha))

    c = cos(-omega)
    s = sin(-omega)
    M_omega = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    #Multiply matrices in the reversed order
    M = np.dot(M_phi, np.dot(M_kappa, M_omega))

    return M;

def taitbryan(phix, phiy, phiz):
    
    T = np.array([[0,0,-1],[0,1,0],[1,0,0]])
    
    c = np.cos(phix)
    s = np.sin(phix)
    M_phix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    c = np.cos(phiy)
    s = np.sin(phiy)
    M_phiy = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    c = np.cos(phiz)
    s = np.sin(phiz)
    M_phiz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    M = np.dot(M_phiz, np.dot(M_phiy, M_phix))

    return np.dot(T, np.dot(M, T.T))

def euler_from_rotation_matrix(R, convention='YZY'):
    
    first = 'XYZ'.find(convention[0])
    second = 'XYZ'.find(convention[1])
    last = 'XYZ'.find(convention[2])
    
    TB = 1 if first+second+last == 3  else 0
    
    par01 = 1 if (last-second) % 3 == 1 else -1
    par12 = 1 if (second-first) % 3 == 1 else -1
                
    s2 = (1-TB-TB*par12)*R[(last+TB*par12)%3,(last-par12)%3]
    c2 = (TB-(1-TB)*par12)*R[(last+TB*par12)%3,(last+par12)%3]
    p2 = np.arctan2(s2,c2)
    
    R2 = np.array([[np.cos(p2), -np.sin(p2), 0], 
                   [np.sin(p2), np.cos(p2), 0],
                   [0, 0, 1]])
    
    Rp = np.dot(R, R2.T)
    
    s0 = par01*Rp[(first-par01)%3,(first+par01)%3]
    c0 = Rp[second,second]
    p0 = np.arctan2(s0,c0)
    
    s1 = par01*Rp[first,3-first-second]
    c1 = Rp[first,first]
    p1 = np.arctan2(s1,c1)
    
    return p0, p1, p2

def kappa_from_euler(phi, chi, omega, alpha=24*np.pi/180):
    
    theta = np.pi/4
        
    kappa = 2*np.arcsin(np.sin(chi/2)/np.sin(alpha))
    delta = np.pi+np.arctan(np.cos(alpha)*np.tan(kappa/2))

    return -phi+delta+theta, alpha, kappa, omega-delta+theta

def euler_from_kappa(phi, kappa, omega, alpha=24*np.pi/180):
    
    theta = np.pi/4
            
    chi = 2*np.arcsin(np.sin(alpha)*np.sin(kappa/2))
    delta = np.pi+np.arctan(np.cos(alpha)*np.tan(kappa/2))

    return -phi+delta+theta, chi, omega+delta-theta

#===============================================================================================
def z_rotation_matrix(polar=0):
    """Generate a rotation matrix for a polar rotation,
    i.e. a rotation about the x axis."""

    #s and c are temp. variables for sin(x) and cos(x)
    c = cos(polar)
    s = sin(polar)
    M = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return M;

#===============================================================================================
def x_rotation_matrix(polar=0):
    """Generate a rotation matrix for a polar rotation,
    i.e. a rotation about the x axis."""

    #s and c are temp. variables for sin(x) and cos(x)
    c = cos(polar)
    s = sin(polar)
    M = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return M;

#===============================================================================================
def rotation_matrix_around_vector(rotation_axis, angle):
    """Generate a rotation matrix about an arbitrary axis.

    Parameters:
        rotation_axis: 3-element vector giving the x,y,z of the rotation axis.
        angle: angle in radians to rotate around the axis. Right-handed coordinates.

    Taken from http://www.euclideanspace.com/maths/algebra/matrix/orthogonal/rotation/index.htm
    """
    # we make a rotation matrix
    (x,y,z) = rotation_axis.flatten() / vector_length(rotation_axis)

    #This is a normalized rotation matrix
    c = cos(angle)
    s = sin(angle)
    R = np.array([[1 + (1-c)*(x*x-1), -z*s+(1-c)*x*y, y*s+(1-c)*x*z],
        [z*s+(1-c)*x*y, 1 + (1-c)*(y*y-1), -x*s+(1-c)*y*z],
        [-y*s+(1-c)*x*z,  x*s+(1-c)*y*z,  1 + (1-c)*(z*z-1)]])

#    R = np.array([[1 + (1-cos(angle))*(x*x-1), -z*sin(angle)+(1-cos(angle))*x*y, y*sin(angle)+(1-cos(angle))*x*z],
#        [z*sin(angle)+(1-cos(angle))*x*y, 1 + (1-cos(angle))*(y*y-1), -x*sin(angle)+(1-cos(angle))*y*z],
#        [-y*sin(angle)+(1-cos(angle))*x*z,  x*sin(angle)+(1-cos(angle))*y*z,  1 + (1-cos(angle))*(z*z-1)]])

    return R

#===============================================================================================
def az_elev_direction(az, elev):
    """Returns a tuple giving the x,y,z coords of a vector pointing towards the given angles.

    Parameters:
        azimuth (az): defined as 0 towards Z positive (down the beam direction)
        elevation (elev): angle going above the XZ plane towards y positive.
            az and elev can be arbitrarily-shaped arrays.

    Returns:
        (x,y,z), defining normalized vectors pointing in that direction.    
    """

    #Assuming azimuth of zero points to z positive = same direction as incident radiation.
    r2 = cos(elev)
    z=cos(az) * r2
    x=sin(az) * r2

    #Assuming polar angle is 0 when horizontal, positive to y positive:
    y=sin(elev)

    #So this is the direction, as a tuple of coords.
    return (x,y,z)


#===============================================================================================
def nearest_index(value_list, desired):
    """Look for the nearest value in a list, and return the index that corresponds to that.
    
    Parameters:
        value_list: numpy array with the list of values. Should be monotonically increasing.
        desired: the desired value, a scalar.
    Returns:
        index to the list.
    """
    if not isinstance(value_list, np.ndarray): return none
    #Check if it is beyond the start or end elements.
    last_err = np.inf
    if desired < value_list[0]:
        return 0
    if desired > value_list[-1]:
        return value_list.size-1
    for i in range(1, value_list.size):
        err = desired - value_list[i]
        if (err==0) or (last_err >= 0) and (err < 0):
            #Sign switch means we passed it
            if np.abs(last_err) < np.abs(err):
                #The one before was closer
                return i-1
            else:
                return i
        last_err = err
    #Return the last element if we reach here
    return value_list.size-1

#===============================================================================================
def index_evenly_spaced(min, max_array_index, spacing, desired):
    """Calculate the index in an evenly spaced, monotonously increasing array.
        min: first value of array
        max_array_index: highest value the index into the array can be
        spacing: space between elements in array.
        desired: the value we are looking for.

    Returns: the index, or None if out of bounds."""

    index = int(np.round( (desired-min)/spacing ))
    if index < 0:
        return None
    if index >= max_array_index:
        return None
    return index

#===============================================================================================
def index_array_evenly_spaced(min, max_array_index, spacing, desired):
    """Calculate AN ARRAY of indices in an evenly spaced, monotonously increasing array.
        min: first value of array
        max_array_index: highest value the index into the array can be
        spacing: space between elements in array.
        desired: the values we are looking for, as a 1D numpy array.

    Returns: numpy array, 1D with the index; NaN where out of bounds"""

    index = np.round( (desired-min)/spacing )
    #Adjust the too-low, too-high indices
    index[ index < 0 ] = np.nan
    index[ index >= max_array_index ] = np.nan
    return index

#===============================================================================================
#======================== VECTOR AND COLUMN CREATION ===========================================
#===============================================================================================

#===============================================================================================
def vector_length(vector):
    """Return the length of the 3- or n-element vector supplied."""
    return np.sqrt(np.sum(np.square(vector)))

#===============================================================================================
def normalize_vector(vector, length=1.0):
    """Normalize the vector to the given length (default of 1)."""
    return length * (vector / np.sqrt(np.sum(np.square(vector))))

#===============================================================================================
def column(vector):
    """Return a column vector from a list or row vector.
    If the input is a matrix, this does nothing.
    If the input is a row vector, this reshapes it into a column,
    If the input is a list or tuple of x, y, z, a column numpy array is created."""
    if isinstance(vector, list) or isinstance(vector, tuple):
        #Make a column array
        rowvec = np.array(vector)
        #Reshape and return
        return rowvec.reshape(3, -1)
    elif isinstance(vector, np.ndarray):
        if vector.ndim > 1:
            #More than 1 dimension, can't reshape
            return vector
        else:
            #Reshape into column
            return vector.reshape(vector.size, 1)
    else:
        #You fed something stupid here!
        return None


#===============================================================================================
def vector(x,y=None,z=None):
    """Make a vector (1D array) from 3 values, or an array, if y and x are not specified/
    """
    if y is None:
        if isinstance(x, list) or isinstance(x, tuple):
            #Make a column array
            vec = np.array(x)
            #Reshape and return
            return vec.reshape(vec.size)
        elif isinstance(x, np.ndarray):
            return x.reshape(x.size)
    else:
        return np.array([x, y, z])


#========================================================================================================
def vectors_to_matrix(*args):
    """From a list of vectors (numpy arrays), make a 3xN matrix where each column is each vector."""
    n = len(args)
    if n == 0:
        return np.zeros(0)
    else:
        if not isinstance(args[0], np.ndarray):
            raise ValueError("vectors_to_matrix needs numpy arrays as inputs.")
        height = args[0].size
        M = np.zeros( (height, n) )
        for i in xrange(len(args)):
            M[:, i] = args[i].flatten()
        return M

#===============================================================================================
def get_translated_vectors(vectors, translation_vector):
    """Translate a matrix containing X,Y,Z vectors (column-wise)
       vectors: input matrix; vectors are column-wise
       translation_vector: what to add to each coordinate.
    Returns: a matrix with the translated vectors. The original vectors are unchanged"""
    if vectors.ndim == 1:
        #Input is just 1 vector, Make the vector into a column
        columns = 1
        vectors = column(vectors)
    else:
        columns = vectors.shape[1]

    #Make a matrix repeating the vector to add to the original
    veclist = list()
    for i in range(columns):
        veclist.append(translation_vector)
    translation_matrix = np.column_stack(tuple(veclist))
    #Now we simply add the translation matrix to it.
    translated = vectors + translation_matrix
    return translated








#================================================================================
#============================ UNIT TESTING ======================================
#================================================================================
import unittest

#==================================================================
class TestNumpyUtils(unittest.TestCase):
    """Unit test for the numpy_utils module."""
    def setUp(self):
        pass

    def isColumn(self, col):
        return isinstance(col, np.ndarray) and col.shape[0] == 3 and len(col.shape) == 2
    def isVector(self, vec):
        return isinstance(vec, np.ndarray) and len(vec.shape) == 1

    #----------------------------------------------------------------
    def test_column(self):
        """column()"""
        assert self.isColumn(column(  (1,2,3)  )), "Tuple input"
        assert self.isColumn(column(  [1,2,3]  )), "list input"
        assert self.isColumn(column(  np.array([1,2,3])  )), "Array input"
        a = np.arange(0, 10)
        assert self.isColumn(column(  (a+1,a+2,a+3)  )), "Tuple of arrays input"

    #----------------------------------------------------------------
    def test_vector_length(self):
        """vector_length()"""
        a = np.array([2, 0, 0])
        assert vector_length(a) == 2, "Length of 2"

    #----------------------------------------------------------------
    def test_vector(self):
        assert self.isVector(vector(  1,2,3  )), "3 arguments"
        assert self.isVector(vector(  (1,2,3)  )), "Tuple input"
        assert self.isVector(vector(  [1,2,3]  )), "list input"
        assert self.isVector(vector(  np.array([1,2,3])  )), "Array input"
        assert self.isVector(vector( column([1,2,3]) )), "Column input"

    #----------------------------------------------------------------
    def test_vectors_to_matrix(self):
        a = vector(1,2,3)
        b = vector(4,5,6)
        c = vector(7,8,9)
        M = np.array(np.arange(1.0, 10.0, 1.0)).reshape(3,3).transpose()
        res = vectors_to_matrix(a,b,c)
        assert np.all(M == res), "Correct result."
        res = vectors_to_matrix(a,b)
        assert res.shape==(3,2), "Correct shape."
        res = vectors_to_matrix(column(a),b,c)
        assert np.all(M == res), "Handles column vectors."
        #Raises error if fed a non-numpy-array
        self.assertRaises(ValueError, vectors_to_matrix, [1,2,3])

    #----------------------------------------------------------------
    def test_rotation_matrix(self):
        """rotation_matrix()"""
        r = np.identity(3)
        rot = rotation_matrix(0,0,0)
        assert rot.shape == (3,3), "Correct shape"
        assert np.allclose(rot, r), "No rotation one"

        r = np.zeros( (3,3) )
        for x in xrange(3): r[x,x] = [-1, 1, -1][x]
        rot = rotation_matrix(np.pi,0,0)
        assert np.allclose(rot, r), "Phi = pi"
        assert np.allclose(np.dot(rot, rot.transpose()), np.eye(3)), "Rotation matrix is orthonormal: M*M.transpose() == Identity."

        rot = rotation_matrix(np.random.random(), np.random.random(), np.random.random())
        assert np.allclose(np.dot(rot, rot.transpose()), np.eye(3)), "Rotation matrix is orthonormal: M*M.transpose() == Identity."
        rot = rotation_matrix(np.random.random()*10, np.random.random()*10, np.random.random()*10)
        assert np.allclose(np.dot(rot, rot.transpose()), np.eye(3)), "Rotation matrix is orthonormal: M*M.transpose() == Identity."

        vr = np.dot(rotation_matrix(np.pi/2,0,0),  column([1,0,0]) ).flatten()
        assert np.allclose(vr, [0,0,-1.] ), "Phi = +pi/2: +x points to -z now"
        vr = np.dot(rotation_matrix(-np.pi/2,0,0),  column([1,0,0]) ).flatten()
        assert np.allclose(vr, [0,0,+1.]), "Phi = -pi/2: +x points to +z now"
        vr = np.dot(rotation_matrix(0,np.pi/2,0),  column([1,0,0]) ).flatten()
        assert np.allclose(vr, [0,+1.,0]), "Chi = +pi/2: +x points to +y now"
        vr = np.dot(rotation_matrix(0,0,np.pi/2),  column([1,0,0]) ).flatten()
        assert np.allclose(vr, [0,0,-1.] ), "Omega = +pi/2: +x points to -z now"

        assert np.allclose(rotation_matrix(np.pi/2,0,0), np.linalg.inv(rotation_matrix(-np.pi/2,0,0))), "Equivalent rotation matrices"
        assert np.allclose(rotation_matrix(0,0.2,0), np.linalg.inv(rotation_matrix(0,-0.2,0))), "Equivalent rotation matrices"

        r = np.array([ [0,-1,0], [1,0,0], [0,0,1] ])
        rot = rotation_matrix(0,np.pi/2,0)
        assert np.allclose(rot, r), "Chi = pi/2"

    #----------------------------------------------------------------
    def test_az_elev_direction(self):
        """az_elev_direction() in the normal coordinates."""
        ret = az_elev_direction(0.1, 0.2)
        assert isinstance(ret, tuple), "Returns a tuple"
        ret = az_elev_direction(0, 0)
        assert np.allclose(ret, (0,0,1)), "Points towards positive Z at 0 azimuth and 0 polar"
        ret = az_elev_direction(np.pi, 0)
        assert np.allclose(ret, (0,0,-1)), "Points towards -Z at pi azimuth and 0 polar"
        ret = az_elev_direction(0, np.pi/2)
        assert np.allclose(ret, (0,1,0)), "Points towards positive Y at positive polar angle"
        ret = az_elev_direction(0, -np.pi/2)
        assert np.allclose(ret, (0,-1,0)), "Points towards negative Y at negative polar angle"
        ret = az_elev_direction(1.234, np.pi/2)
        assert np.allclose(ret, (0,1,0)), "Points towards positive Y at positive polar angle, at any azimuth"
        az = np.arange(0,1,0.1)
        pol = np.arange(1,2,0.1)
        (x,y,z) = az_elev_direction(az, pol)
        assert isinstance(x, np.ndarray), "Returns a tuple of arrays"
        assert x.size == 10, "Returns a tuple of arrays of correct size"
        expect = np.fromstring(" 0.          0.04528405  0.07198937  0.07905131  0.06618832  0.03391322 -0.01648729 -0.0830039  -0.16298481 -0.2532414", sep=" ")
        assert np.allclose(x,expect), "Values match."

    #----------------------------------------------------------------
    def test_normalize_vector(self): 
        """normalize_vector()"""
        vec = np.array([1, 2, 3])
        assert np.allclose(vector_length( normalize_vector(vec) ), 1.0), "Normalized to length 1.0"
        assert np.allclose(vector_length( normalize_vector(vec, 15.2) ),  15.2), "Normalized to length 15.2"
        vec = np.array([1, 1, 0])
        assert np.allclose(normalize_vector(vec), np.sqrt(2)/2*np.array([1,1,0])), "Checked direction"

    #----------------------------------------------------------------
    def try_angles(self, val):
        M = rotation_matrix(val[0], val[1], val[2])
        res = angles_from_rotation_matrix(M)
        res_M = rotation_matrix(res[0], res[1], res[2])
#        print "asked for", np.round(np.rad2deg(val),1)," got ", np.round(np.rad2deg(res),1)
        assert np.allclose(M, res_M), "Rotation matrix finding for %s" % np.array(val)

    def test_angles_from_rotation_matrix(self):
        self.try_angles( np.deg2rad([10, -10, 20]) )
        self.try_angles( np.deg2rad([175, -10, 45]) )

        #Try some small angles
        self.try_angles( np.deg2rad([10, 0.1, 0.2]) )
        self.try_angles( np.deg2rad([0.01, -0.05, 0.02]) )
        self.try_angles( np.deg2rad([0.01, 0.01, 0.02]) )

        #Test all the possible quadrants
        for phi in np.arange(-1, 1, 0.125)*2*np.pi:
            for chi in np.arange(-1, 1, 0.25)*np.pi:
                for omega in np.arange(-1, 1, 0.125)*2*np.pi:
                    self.try_angles( (phi, chi, omega) )

    def test_equivalent_goniometer_angles(self):
        """rotation_matrix(phi,chi,omega) == rotation_matrix(phi-np.pi, -chi, omega-np.pi)"""
        for phi in np.arange(-1, 1, 0.125)*2*np.pi:
            for chi in np.arange(-1, 1, 0.25)*np.pi:
                for omega in np.arange(-1, 1, 0.125)*2*np.pi:
                    M1 = rotation_matrix(phi,chi,omega)
                    M2 = rotation_matrix(phi-np.pi, -chi, omega-np.pi)
                    assert np.allclose(M1, M2), "Equivalent goniometer angles 1 (phi-np.pi, -chi, omega-np.pi)."
#                    M2 = rotation_matrix(phi+2*np.pi, chi, omega)
#                    assert np.allclose(M1, M2), "Equivalent goniometer angles 2."

#---------------------------------------------------------------------
# if __name__ == "__main__":
#     unittest.main()
