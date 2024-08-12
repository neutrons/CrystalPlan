""" Goniometer Module:

Class hierarchy for various sample orientation goniometers.
 - Goniometer: base class, an unrestricted goniometer.
    - LimitedGoniometer: a goniometer that has limits in what it can achieve. Virtual.
        - TopazInHouseGoniometer: 3-legged model
        - TopazAmbientGoniometer: Chi fixed at 45 deg
Performs transformation from sample rotation coordinates to calculate the goniometer motor positions, and vice-versa.
Calculates allowable angles (phi, chi, omega).
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import copy
import time
import warnings
import string
from string import replace, strip, find
import numpy as np
from numpy import array, sin, cos, pi, dot
import scipy.optimize
import weave

#--- Model Imports ---
import numpy_utils
from numpy_utils import column, vector_length, rotation_matrix, get_translated_vectors, nearest_index
import utils

#--- Traits Imports ---
from traits.api import HasTraits,Int,Float,Str,String,Property,Bool, List, Tuple, Array, Enum
from traitsui.api import View,Item,Group,Label,Heading, Spring, Handler, TupleEditor, TabularEditor, ArrayEditor, TextEditor, CodeEditor, ListEditor
from traitsui.menu import OKButton, CancelButton,RevertButton
from traitsui.menu import Menu, Action, Separator


#---- For plotting ----
if __name__=="__main__":
    import mayavi.mlab as mlab
    import pylab

#===============================================================================================
#Constants defining the positions in the matrix
MOUNT_A = 0
MOUNT_B = 1
MOUNT_C = 2
#Vectors are column-wise, therefore these are the indices of the ROWS to get these coordinates
COORD_X = 0
COORD_Y = 1
COORD_Z = 2


#===============================================================================================
def csv_friendly_string(input):
    """Clean-up the input string to make it acceptable for csv output."""
    #Convert to string and strip of whitespace
    s = strip(str(input))
    #Newlines are bad
    s = replace(s, '\n', " ")
    s = replace(s, '\r', " ")
    #Replace double quotes with double-double
    s = replace(s, '"', '""')

    #Add quotes if needed
    found = [(find(s,x)>0) for x in ' ",']
    if any(found):
        s = '"' + s + '"'

#    #Add quotes if needed
#    for x in s:
#        if not x in (string.digits + string.ascii_letters):
#            #Found a non-alphanumeric character
#            s = '"' + s + '"'
#            break
            
    return s

#===============================================================================================
def csv_line(input_list):
    """Make a line acceptable for CSV from an input list of various types.
    Line has a newline character at the end"""
    good_strings = [csv_friendly_string(x) for x in input_list]
    return string.join(good_strings, ",") + "\n"




#========================================================================================================
#========================================================================================================
#========================================================================================================
class AngleInfo(HasTraits):
    """Class holds some relevant information about an angle used in the sample orientation."""
    name = String()
    type = String(desc='type of value being set.')
    units = String(label='Internal units', desc='the units used internally in computations.')
    friendly_units = String(desc='the "friendly" units displayed in the GUI.')
    conversion = Float(label='Internal to friendly unit conversion factor', desc='that multiplying internal units by this value gives you the friendly units')
    das_units = String(label='DAS units', desc='the units required by the DAS group (used in the output CSV file).')
    das_conversion = Float(label='Internal to DAS unit conversion factor', desc='that multiplying internal units by this value gives you the DAS units')
    friendly_range = List(Float, label='Range of values in GUI (friendly units)', desc='two numbers giving the minimum and maximum values to show in GUI, for the sliders for example.')

    view=View(
        Item('name'),Item('type'),
            Group( Item('units'),Item('friendly_units'),
                    Item('conversion'),Item('das_units'),  Item('das_conversion'), label='Units'),
            Group( Item('friendly_range'), label="Ranges"),
        buttons=[OKButton, CancelButton]
        )


    def __init__(self, name, type="angle", units="rad", friendly_units="deg", conversion=np.deg2rad(1),
                    friendly_range=[-180, 180], random_range=None,
                    das_units="deg", das_conversion=np.deg2rad(1)):
        """Constructor.

        Parameters:
            name: Name of the angle
            type: Type of angle (or other thing)
            units: Units used internally
            friendly_units: Friendly units, to be displayed to users
            conversion: Conversion factor from internal units to friendly units (multiply the internal by THIS to get friendly)
            friendly_range: Range to use, in friendly units; for sliders and such.
            random_range: Randomization range: when generating an angle at random (in a genetic algorithm, for example,
                 use this range to initialize population.  In internal units. Will be the same as friendly_range, but converted to "units".
            das_conversion: Units required by the DAS group to be used in the output CSV file
            das_units: Conversion factor from internal units to DAS units (multiply the internal by THIS to get DAS units)
        """
        self.name = name
        self.type = type
        self.units = units
        self.friendly_units = friendly_units
        if conversion != 0:
            self.conversion = conversion
        self.friendly_range = friendly_range
        self.random_range = list(np.array(friendly_range) * self.conversion)
        self.das_conversion = das_conversion
        self.das_units = das_units

    #========================================================================================================
    def __eq__(self, other):
        return utils.equal_objects(self, other)

    def friendly_to_internal(self, value):
        """Convert a friendly unit angle value to an internal value."""
        return value * self.conversion

    def internal_to_friendly(self, value):
        """Convert an internal angle value to a friendly unit one."""
        return value / self.conversion

    def internal_to_das(self, value):
        """Convert an internal angle value to a DAS group-required unit one."""
        return value / self.das_conversion

    def pretty_print(self, value, add_unit=False):
        """Return a string with a pretty printed value.

        Parameters:
            value: value in internal units.
            add_unit: also show the unit.
        """
        s = "%.1f" % self.internal_to_friendly(value)
        if add_unit: s += " " + self.friendly_units
        return s

    def get_random(self):
        """Return a random angle within the specified range of this angle."""
        self.random_range = list(np.array(self.friendly_range) * self.conversion)
        return np.random.uniform(self.random_range[0], self.random_range[1], 1)[0]

    def __str__(self):
        return "%s, %s, ranges from %.1f to %.1f %s" % (self.name, self.type, self.friendly_range[0], self.friendly_range[1], self.friendly_units)
    def __repr__(self):
        return self.__str__()

    def is_angle_valid(self, angle):
        """ Return true if the angle is within the valid random range of this
        AngleInfo.

        Parameters:
            angle: angle in INTERNAL units."""

        self.random_range = list(np.array(self.friendly_range) * self.conversion)
        if (angle < self.random_range[0]) or (angle > self.random_range[1]):
            return False
        else:
            return True








#===============================================================================================
#===============================================================================================
#===============================================================================================
class Goniometer(HasTraits):
    """Base class for goniometers. Will be overridden by specific classes for different types
    of instruments."""

    #Some info about the goniometer
    name = String("Simple Goniometer")
    description = String("Simple universal goniometer with no restrictions on movement.")
    wavelength_control = Bool(False, desc="if the goniometer also controls the measurement wavelength.")
    gonio_angles = List(AngleInfo, label='Goniometer angles', desc="the list of goniometer angles.")
    wl_angles = List(AngleInfo, label='Wavelength control parameters', desc="the list of wavelength control parameters.", rows_trait=1)

    angles_desc = Property(String, depends_on=["gonio_angles", "wl_angles", "wavelength_bandwidth"], label='Description of Angles:', desc="a description of the angles/other positions controlled by the goniometer")
    def _get_angles_desc(self):
        return self.get_angles_description()

    wavelength_bandwidth = Property(Float, desc="the bandwidth of measurement wavelength, in angstroms. Changing this value changes the allowable range for WL_center.")
    _wavelength_bandwidth = 0.6
    def _get_wavelength_bandwidth(self):
        return self._wavelength_bandwidth
    def _set_wavelength_bandwidth(self, value):
        self._wavelength_bandwidth = value
        #Set the range of bandwidth center so that it can't go too low or too high, using the
        #   _wavelength_minimum and _wavelength_maximum
        if hasattr(self, 'wl_angles'):
            if len(self.wl_angles) > 0:
                self.wl_angles[0].friendly_range[0] = self._wavelength_minimum + value/2.0
                self.wl_angles[0].random_range[0] = self._wavelength_minimum + value/2.0
                self.wl_angles[0].friendly_range[1] = self._wavelength_maximum - value/2.0
                self.wl_angles[0].random_range[1] = self._wavelength_maximum - value/2.0

    wavelength_minimum = Property(Float, desc="the minimum wavelength that can be reached, in angstroms. Changing this value changes the allowable range for WL_center.")
    _wavelength_minimum = 0.1
    def _get_wavelength_minimum(self):
        return self._wavelength_minimum
    def _set_wavelength_minimum(self, value):
        self._wavelength_minimum = value
        #This takes care of setting the ranges correctly.
        self._set_wavelength_bandwidth(self._wavelength_bandwidth)

    wavelength_maximum = Property(Float, desc="the maximum wavelength that can be reached, in angstroms. Changing this value changes the allowable range for WL_center.")
    _wavelength_maximum = 3.7
    def _get_wavelength_maximum(self):
        return self._wavelength_maximum
    def _set_wavelength_maximum(self, value):
        self._wavelength_maximum = value
        #This takes care of setting the ranges correctly.
        self._set_wavelength_bandwidth(self._wavelength_bandwidth)


    view = View(
        Item('name'), Item('description'),
        Item('wavelength_control'),
        Item('wavelength_bandwidth', visible_when="wavelength_control"),        Item('wavelength_minimum', visible_when="wavelength_control"),        Item('wavelength_maximum', visible_when="wavelength_control"),
        Item('angles_desc', style='readonly')
        )



    #-------------------------------------------------------------------------
    def __eq__(self, other):
        """Return True if the contents of self are equal to other."""
        return (self.name == other.name) and (self.wavelength_control == other.wavelength_control) \
                and (self.gonio_angles == other.gonio_angles) and (self.wl_angles == other.wl_angles) \
                and (self.wavelength_minimum == other.wavelength_minimum) \
                and (self.wavelength_maximum == other.wavelength_maximum) \
                and (self.wavelength_bandwidth == other.wavelength_bandwidth)

    def __ne__(self,other):
        return not self.__eq__(other)
    
    #-------------------------------------------------------------------------
    def __init__(self, wavelength_control=False):
        """Constructor.

        Parameters:
            wavelength_control : bool, have the goniometer control wavelength too."""
        #Make the angle info object
        self.gonio_angles = [AngleInfo('Phi'), AngleInfo('Chi'), AngleInfo('Omega')]
        #Do we also change wavelengths?
        self.wavelength_control = wavelength_control
        #Create the angle info for it, but don't necessarily use it.
        self.wl_angles = [
            AngleInfo('WL_Center', type="wavelength", units="ang", friendly_units="ang",
                      conversion=1, friendly_range=[1.0, 10], random_range=[1.7, 10],
                      das_units="ang", das_conversion=1.0)
            ]
        #Bandwidth of detection, in angstroms
        self.wavelength_minimum = 0.1
        self.wavelength_maximum = 3.7
        self.wavelength_bandwidth = 0.6

    #========================================================================================================
    def __eq__(self, other):
        return utils.equal_objects(self, other)

    #-------------------------------------------------------------------------
    def are_angles_allowed(self, angles, return_reason=False):
        """Calculate whether the given angles can be reached with the goniometer.
        Will be overridden by subclasses as needed.

        Parameters:
            angles: list of angles of the goniometer (typically phi, chi, omega).
            return_reason: do you want to return the reason (as string)?

        Returns:
            allowed: True if the angle provided can be reached.
            reason: string describing the reason they cannot be reached. Empty if the angles are allowed.
        """
        #The base goniometer can reach all angles.
        if return_reason:
            return (True, "")
        else:
            return True


    #-------------------------------------------------------------------------
    def get_angles(self):
        """Return a list of AngleInfo objects describing the degrees of
        freedom of the goniometer."""
        if self.wavelength_control:
            return self.gonio_angles + self.wl_angles
        else:
            return self.gonio_angles

    def set_angles(self, value):
        raise NotImplementedError("Do not set the Goniometer.angles attribute directly!")

    angles = property(get_angles, set_angles)

    #-------------------------------------------------------------------------
    def get_angles_description(self):
        """Return a string describing each angle for the goniometer."""
        s = []
        for ang in self.gonio_angles:
            s.append("%s" % ang)
        if self.wavelength_control:
            for ang in self.wl_angles:
                s.append("%s" % ang)
        return "\n".join(s)


    #-------------------------------------------------------------------------------
    def get_wl_input(self):
        return None
    def get_wavelength_range(self, angles):
        """Return (wl_min, wl_max) from the list of angles, or (None, None) if not specified.
        """
        if self.wavelength_control:
            wl_center = angles[len(self.gonio_angles)]
            wl_min = wl_center - self.wavelength_bandwidth/2.
            wl_max = wl_center + self.wavelength_bandwidth/2.
            if wl_min < 1e-3: wl_min = 1e-3
            if wl_max < 1e-3: wl_max = 1e-3
            return (wl_min, wl_max)
        else:
            return (None, None)



    #-------------------------------------------------------------------------------
    def make_q_rot_matrix(self, angles):
        """Generate the necessary rotation matrix for use in the getq method.
        The q rotation matrix corresponds to the opposite (negative) angles that
        are the sample rotation angles.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        #For other instruments, this method may be different.
        (phi, chi, omega) = angles[0:3]

        #In Q space, detector coverage rotates OPPOSITE to what the real space rotation is.
        #Because that is where the detectors and incident beam go, AS SEEN BY THE SAMPLE.

        #So wee need to invert the sample orientation matrix to find the one that will apply to the Q vector.
        return numpy_utils.opposite_rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------------
    def make_sample_rot_matrix(self, angles):
        """Generate the sample rotation matrix, from the given sample orientation angles.
        Unlike make_q_rot_matrix(), the direct angles are used here.
        This matrix will be used to calculate the scattering angle of specific reflections.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        (phi, chi, omega) = angles[0:3]
        return numpy_utils.rotation_matrix(phi, chi, omega)
    

    #-------------------------------------------------------------------------
    def calculate_angles_to_rotate_vector(self, starting_vec, ending_vec, starting_angles=None, search_method=0):
        """Calculate a set of sample orientation angles that rotate a single vector.
        TRY to return a sample orientation that is achievable by the goniometer.

        Parameters:
            starting_vec: starting vector of interest.
            ending_vec: resulting vector after desired rotation.
            starting_angles: optional, list of angles from which to start the search.

        Return:
            angles: list of angles found.
        """
        #Find the starting rotation matrix
        if not starting_angles is None:
            (phi, chi, omega) = starting_angles[0:3]
            starting_rot_matrix = numpy_utils.rotation_matrix(phi, chi, omega)
            #Rotate the starting vector
            starting_vec = np.dot(starting_rot_matrix, column(starting_vec)).flatten()

        #Find the rotation matrix that satisfies ending_vec = R . starting_vec

        #The cross product of q0 X q_over_a gives a rotation axis to use
        rotation_axis = np.cross(starting_vec, ending_vec)

        #Now we find the rotation angle about that axis that puts q0 on q_over_a
        angle = np.arccos( np.dot(starting_vec, ending_vec) / (vector_length(starting_vec)*vector_length(ending_vec)))

        #Make the rot. matrix
        R = numpy_utils.rotation_matrix_around_vector(rotation_axis, angle)

        if not starting_angles is None:
            #The final rotation we want is starting_rot_matrix 1st; R second.
            # So this is the resulting matrix
            R = np.dot(R, starting_rot_matrix)

        #The function finds some angles that work
        angles = numpy_utils.angles_from_rotation_matrix(R)

        #Position is always allowed
        return (angles)



    #========================================================================================================
    def get_sample_orientation_to_get_beam(self, beam_wanted, hkl, ub_matrix, starting_angles=None, search_method=0):
        """Find the sample orientation angles that will result in a scattered beam
        going in the desired direction.

        Parameters:
            beam_wanted: vector containing a NORMALIZED vector of the DESIRED scattered beam direction.
            hkl: vector containing the h,k,l reflection of interest.
            ub_matrix: for the current sample and mounting. B converts hkl to q-vector, and U orients.
            starting_angles: optional, list of angles from which to start the search.
                e.g. we want the beam to move a few degrees away from this starting sample orientation.
            search_method: option for the search method.

        Returns:
            angles: list of angles found
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
        q0 = np.dot(ub_matrix, hkl).flatten()

        #Since the rotation does not change length, norm(q0) = norm(q_over_a),
        #   meaning that a = norm(q)/ norm(q_over_a)
        a = vector_length(q0) / vector_length(q_over_a)
        #And a's definition is such that:
        wavelength = 2*pi/a

        #So lets get q directly.
        q_rotated = q_over_a * a

        #Separate function finds the phi, [chi], omega angles.
        (angles) = self.calculate_angles_to_rotate_vector(q0, q_rotated, starting_angles, search_method)
        if self.wavelength_control and not (angles is None):
            #Need to add a wavelength angle
            wl_center = wavelength
            wl_center = max(wl_center, self.wl_angles[0].friendly_range[0]) #Lowest bound
            wl_center = min(wl_center, self.wl_angles[0].friendly_range[1]) #Higher bound
            angles += [wl_center]

        return (angles, wavelength)


    #===============================================================================================
    def csv_make_header(self, fileobj, title, comment=""):
        """Make the header text of the motor positions CSV file. This is general for universal
        goniometers, but can be subclassed by specific ones.

        Parameters:
        -----------
            fileobj: an already open, writable file object.
        """
        #Line of header info
        fileobj.write(csv_line( ['Comment'] + [x.name.lower() for x in self.angles] + ['Wait For', 'Value'] ) )


    #===============================================================================================
    def csv_add_position(self, fileobj, angle_values, count_for, count_value, comment):
        """Add a line to an existing CSV file containing the motor positions, etc. for
        that part of the experiment.

        Parameters:
            angles: list of angles in the sample orientation.
            count_for, count_value: stopping criterion
        """
        #Calculate if its allowed
        (allowed, reason) = self.are_angles_allowed(angle_values, return_reason=True)
        #Convert from internal to DAS units.
        das_angles = [self.angles[i].internal_to_das(angle_values[i]) for i in xrange(len(self.angles))]
        stopping_criterion = count_for
        if stopping_criterion == "runtime":
            stopping_criterion = "seconds"
        if not allowed:
            #Can't reach this position
            fileobj.write("#"" ----- ERROR! This sample orientation could not be achieved with the goniometer, because of '%s'. THE FOLLOWING LINE HAS BEEN COMMENTED OUT ------ ""\n" % reason )
            fileobj.write('#' + csv_line( [comment]+ das_angles + [stopping_criterion, count_value] ) )
        else:
            #They are okay
            fileobj.write(csv_line( [comment] + das_angles + [stopping_criterion, count_value] ) )

    #===============================================================================================
    def get_phi_chi_omega(self, angles):
        """Given a list of angles (which may have more or less angles depending on goniometer type),
        return the equivalent (phi, chi, omega) in radians."""
        (phi, chi, omega) = angles[0:3]
        return (phi, chi, omega)



#===============================================================================================
#===============================================================================================
#===============================================================================================
class LimitedGoniometer(Goniometer):
    """Class for goniometers where there are limits to the range of phi, chi, and/or omega
    values. Holds general-purpose code for finding valid sample orientations."""
    
    name = String("Limited Goniometer")
    description = String("Goniometer with 3 degrees of freedom but restrictions to allowed angles.")

    #-------------------------------------------------------------------------
    def __init__(self, wavelength_control=False, *args, **kwargs):
        Goniometer.__init__(self, wavelength_control, *args, **kwargs)

    #-------------------------------------------------------------------------
    def are_angles_allowed(self, angles, return_reason=False):
        """Calculate whether the given angles can be reached with the goniometer.
        Will be overridden by subclasses as needed.

        Parameters:
            angles: list of angles of the goniometer (typically phi, chi, omega).
            return_reason: do you want to return the reason (as string)?

        Returns:
            allowed: True if the angle provided can be reached.
            reason: string describing the reason they cannot be reached. Empty if the angles are allowed.
        """
        ret = True
        reason = ""
        all_angle_infos = self.get_angles() 

        for i in xrange(len(angles)):
            #@type AngleInfo ai
            ai = all_angle_infos[i]
            angle = angles[i]
            if angle < ai.random_range[0] or angle > ai.random_range[1]:
                ret = False
                reason += ai.name + " outside limits."

        #The base goniometer can reach all angles.
        if return_reason:
            return (ret, reason)
        else:
            return ret


    #-------------------------------------------------------------------------
    def get_fitness_function_c_code(self):
        #C code for the fitness of phi,chi, omega.
        # OVERWRITE THIS FOR SUBCLASSES!
        #   Don't change the # of parameters - the search code always gives you phi, chi, omega.
        # This sample fitness value makes phi less important (since it normally has full freedom)

        args = []
        for i in xrange(3):
            for j in xrange(2):
                args.append(self.gonio_angles[i].random_range[j])
        args = tuple(args)

        s = """
        FLOAT fitness_function(FLOAT phi, FLOAT chi, FLOAT omega)
        {
            FLOAT phi_min = %f;
            FLOAT phi_max = %f;
            FLOAT chi_min = %f;
            FLOAT chi_max = %f;
            FLOAT omega_min = %f;
            FLOAT omega_max = %f;

            FLOAT phi_mid = (phi_min + phi_max) / 2;
            FLOAT chi_mid = (chi_min + chi_max) / 2;
            FLOAT omega_mid = (omega_min + omega_max) / 2;

            FLOAT fitness =  absolute(chi - chi_mid) + absolute(omega - omega_mid) + absolute(phi - phi_mid);

            // Big penalties for being out of the range
            if (phi < phi_min) fitness += (phi_min - phi) * 1.0;
            if (phi > phi_max) fitness += (phi - phi_max) * 1.0;
            if (chi < chi_min) fitness += (chi_min - chi) * 1.0;
            if (chi > chi_max) fitness += (chi - chi_max) * 1.0;
            if (omega < omega_min) fitness += (omega_min - omega) * 1.0;
            if (omega > omega_max) fitness += (omega - omega_max) * 1.0;

            // if (phi < phi_min || phi > phi_max) fitness += 10;
            // if (chi < chi_min || chi > chi_max) fitness += 10;
            // if (omega < omega_min || omega > omega_max) fitness += 10;

            return fitness;
        }""" % (args)
        return s



    #-------------------------------------------------------------------------
    def _angle_fitness_python(self, rot_angle, initial_rotation_matrix, ending_vec, starting_vec):
        """Function for optimization of angles to get the right sample orientation angles.
        We start with a rotation matrix, which we rotate around our desired vector.
        The fitness of it is how achievable the angle is with this goniometer.

        Parameters:
            rot_angle: rotation of initial_rotation_matrix we are making, around the axis ending_vec
            initial_rotation_matrix: initial rotation matrix
            ending_vec: goal vector
            starting_vec: start vector

        Return:
            fitness: metric of fitness to minimize
            best_angles: sample orientation angles corresponding to the given fitness
        """
        #The scipy.optimize routines use an array
        scipy_optimize = False
        if isinstance(rot_angle, np.ndarray):
            scipy_optimize = True
            rot_angle = rot_angle[0]

        #Rotation matrix for these angles
        extra_rotation_matrix = numpy_utils.rotation_matrix_around_vector(ending_vec, rot_angle)
        total_rot_matrix = np.dot(extra_rotation_matrix, initial_rotation_matrix)

        #Now we find the 3 angles
        angles1 = list(numpy_utils.angles_from_rotation_matrix(total_rot_matrix))
        (phi, chi, omega) = angles1

        #Test that the resulting matrix is still OK
        if False:
            result_vec = np.dot(rotation_matrix(phi,chi,omega), column(starting_vec)).flatten()
            assert np.allclose( result_vec, ending_vec, atol=1e-5), "rotated rotation matrix still makes the correct rotation. Expected %s, got %s" % (ending_vec, result_vec)


        #List of fitnesses
        fitnesses = []

        def fitness_calc(phi, chi, omega):
            #And we build a fitness based on these. smaller = better

            #Make all the angles closer to 0
            if phi > pi: phi -= 2*pi
            if chi > pi: chi -= 2*pi
            if omega > pi: omega -= 2*pi
            if phi < -pi: phi += 2*pi
            if chi < -pi: chi += 2*pi
            if omega < -pi: omega += 2*pi

            # Phi is unconstrained. But we prefer it small, to a much lesser degree. Make it a tie-breaker
            fitness = np.abs(chi) + np.abs(omega) + np.abs(phi)/10000

            #For speed-up, we don't check ALL the time if it is allowed
            if abs(chi) > 0.25 or abs(omega) > 0.25: #(about 14 degrees)
                allowed = False
            elif abs(chi) < 0.08 and abs(omega) < 0.08: #(less than about 5 degrees)
                allowed = True
            else:
                #Check it
                allowed = self.are_angles_allowed([phi, chi, omega])

            #Not allowed stuff = much worse fitness
            if not allowed:
                fitness += 10

            return (fitness, (phi, chi, omega))

        fitnesses.append( fitness_calc(phi, chi, omega) )
        # (phi-pi, -chi, omega-pi) is always equivalent
        fitnesses.append( fitness_calc(phi-pi, -chi, omega-pi) )

        #The minimum (fitness) is the best fitness
        (fitness, best_angles) = min(fitnesses)

        #The best option is returned
        if scipy_optimize:
            return fitness
        else:
            return (fitness, best_angles)



    #-------------------------------------------------------------------------
    def _angle_fitness_brute(self, rot_angle_list, initial_rotation_matrix, ending_vec, starting_vec):
        """Function for optimization of angles to get the right sample orientation angles.
        We start with a rotation matrix, which we rotate around our desired vector.
        The fitness of it is how achievable the angle is with this goniometer.

        Parameters:
            rot_angle_list: numpy array of the rotation of initial_rotation_matrix we are making, around the axis ending_vec
            initial_rotation_matrix: initial rotation matrix
            ending_vec: goal vector. MUST BE NORMALIZED TO 1
            starting_vec: start vector

        Return:
            fitness: metric of fitness to minimize
            best_angles: sample orientation angles corresponding to the given fitness
        """

        #General purpose support code.
        support = """
        #define PI 3.14159265358979323846264338327950288
        #define FLOAT double
        FLOAT absolute(FLOAT value)
        {
            if (value<0) {return -value; }
            else { return value; }
        }
        """
        
        #Add the function for the fitness
        support += self.get_fitness_function_c_code()

        code = """
        FLOAT rot_angle;
        int angle_num;
        for (angle_num=0;  angle_num < Nrot_angle_list[0]; angle_num++)
        {
            rot_angle = ROT_ANGLE_LIST1(angle_num);
            //printf("angle of %e\\n", rot_angle);
            // --- Make the rotation matrix around the ending_vec ----
            FLOAT c = cos(rot_angle);
            FLOAT s = sin(rot_angle);
            FLOAT x,y,z;
            x = ending_vec[0];
            y = ending_vec[1];
            z = ending_vec[2];

            FLOAT extra_rotation_matrix[3][3] = {
            {1 + (1-c)*(x*x-1), -z*s+(1-c)*x*y, y*s+(1-c)*x*z},
            {z*s+(1-c)*x*y, 1 + (1-c)*(y*y-1), -x*s+(1-c)*y*z},
            {-y*s+(1-c)*x*z,  x*s+(1-c)*y*z,  1 + (1-c)*(z*z-1)}
            };

            // Do matrix multiplication
            FLOAT total_rot_matrix[3][3];

            int i,j,k;
            for (i=0; i<3; i++)
                for (j=0; j<3; j++)
                {
                    total_rot_matrix[i][j] = 0;
                    for (k=0; k<3; k++)
                    {
                        total_rot_matrix[i][j] += extra_rotation_matrix[i][k] * INITIAL_ROTATION_MATRIX2(k,j);
                    }
                    // printf("%f, ", total_rot_matrix[i][j]);
                }

            //-------- Now we find angles_from_rotation_matrix() -----------
            FLOAT chi, phi, omega;

            //#Let's make 3 vectors describing XYZ after rotations
            FLOAT ux = total_rot_matrix[0][0];
            FLOAT uy = total_rot_matrix[1][0];
            FLOAT uz = total_rot_matrix[2][0];
            FLOAT vx = total_rot_matrix[0][1];
            FLOAT vy = total_rot_matrix[1][1];
            FLOAT vz = total_rot_matrix[2][1];
            FLOAT nx = total_rot_matrix[0][2];
            FLOAT ny = total_rot_matrix[1][2];
            FLOAT nz = total_rot_matrix[2][2];

            //#is v.y vertical?
            if (absolute(vy) < 1e-8)
            {
                //#Chi rotation is 0, so we just have a rotation about y
                chi = 0.0;
                phi = atan2(nx, nz);
                omega = 0.0;
            }            
            else if (absolute(vy+1) < 1e-8)
            {
                //#Chi rotation is 180 degrees
                chi = PI;
                phi = -atan2(nx, nz);
                if (phi==-PI) phi=PI;
                omega = 0.0;
            }
            else
            {
                //#General case
                phi = atan2(ny, uy);
                chi = acos(vy);
                omega = atan2(vz, -vx);
            }

            FLOAT fitness;
            FLOAT old_phi = phi;
            FLOAT old_chi = chi;
            FLOAT old_omega = omega;

            // Try the original angles
            fitness = fitness_function(phi, chi, omega);
            fitnesses.append(fitness);
            phi_list.append(phi);
            chi_list.append(chi);
            omega_list.append(omega);

            //Make angles closer to 0
            if (phi > PI) phi -= 2*PI;
            if (chi > PI) chi -= 2*PI;
            if (omega > PI) omega -= 2*PI;
            if (phi < -PI) phi += 2*PI;
            if (chi < -PI) chi += 2*PI;
            if (omega < -PI) omega += 2*PI;
            fitness = fitness_function(phi, chi, omega);
            fitnesses.append(fitness);
            phi_list.append(phi);
            chi_list.append(chi);
            omega_list.append(omega);

            //(phi-pi, -chi, omega-pi) is always equivalent
            phi = old_phi-PI;
            chi = -old_chi;
            omega = old_omega-PI;
            if (phi > PI) phi -= 2*PI;
            if (chi > PI) chi -= 2*PI;
            if (omega > PI) omega -= 2*PI;
            if (phi < -PI) phi += 2*PI;
            if (chi < -PI) chi += 2*PI;
            if (omega < -PI) omega += 2*PI;
            fitness = fitness_function(phi, chi, omega);
            fitnesses.append(fitness);
            phi_list.append(phi);
            chi_list.append(chi);
            omega_list.append(omega);
        }"""
        #Workaround for bug in weave, where it ignores any changes in the support code.
        code += "\n\n // " + self.__class__.__name__ + "\n"
        code += "/* " + self.get_fitness_function_c_code() + " */"

        #List of fitnesses
        fitnesses = []
        chi_list = []
        phi_list = []
        omega_list = []
       
        #Prepare variables, run the C code
        varlist = ['rot_angle_list', 'ending_vec', 'initial_rotation_matrix', 'fitnesses', 'chi_list', 'phi_list', 'omega_list']
        ret = weave.inline(code, varlist, compiler='gcc', support_code=support)

        #Test that the resulting matrix is still OK
        if False:
            for index in xrange(len(fitnesses)):
                chi = chi_list[index]
                phi = phi_list[index]
                omega = omega_list[index]
                result_vec = np.dot(rotation_matrix(phi,chi,omega), column(starting_vec)).flatten()
                assert np.allclose( result_vec, ending_vec, atol=1e-3), "rotated rotation matrix still makes the correct rotation. Expected %s, got %s" % (ending_vec, result_vec)

        #The minimum (fitness) is the best fitness.
        # There are 3 output values per input angle
        index = np.argmin(fitnesses)
        fitness = fitnesses[index]
        chi = chi_list[index]
        phi = phi_list[index]
        omega = omega_list[index]

        return (rot_angle_list[index/3], (phi, chi, omega))


    #-------------------------------------------------------------------------
    def calculate_angles_to_rotate_vector(self, starting_vec, ending_vec, starting_angles=None, search_method=0):
        """Calculate a set of sample orientation angles that rotate a single vector.
        TRY to return a sample orientation that is achievable by the goniometer.

        Parameters:
            starting_vec: starting vector of interest.
            ending_vec: resulting vector after desired rotation.
            starting_angles: optional, list of angles from which to start the search.
            search_method: 0 for default search (semi-brute-force), 1 to use scipy.optimize

        Return:
            best_angles: list of the 3 angles phi, chi, omega found. None if invalid inputs were given
        """
        # print "starting_vec, ending_vec", starting_vec, ending_vec

        # We want to find a rotation matrix R
        # R puts starting_vec onto ending_vec
        # But R has the freedom to rotate all the other axes around ending_vec - all
        #   of these are equally valid.

        if np.allclose(vector_length(starting_vec), 0) or np.allclose(vector_length(ending_vec), 0):
            return None

        #Normalize our vectors
        starting_vec = starting_vec/vector_length(starting_vec)
        ending_vec = ending_vec/vector_length(ending_vec)

        #Find an initial rotation matrix.
        # We'll rotate around the cross-product of start x end, staying in the plane defined by these vectors
        rotation_axis = np.cross(starting_vec, ending_vec)
        #TODO: check for too-close vectors to get a valid cross-product
        angle = np.arccos( np.dot(starting_vec, ending_vec) )
        initial_R = numpy_utils.rotation_matrix_around_vector(rotation_axis, angle)

        result_vec = np.dot(initial_R, column(starting_vec)).flatten()
        #Check that the matrices match, but not if all are NaN
        #if not np.any(np.isnan(result_vec) and np.isnan(ending_vec)):
        if not np.any(np.isnan(result_vec)):
            assert np.allclose( result_vec, ending_vec), "initial rotation matrix makes the correct rotation. Got %s, expected %s" % ( result_vec, ending_vec)


        def optimize(start, stop, step):
            """Routine to optimize by brute force"""
            #Go through every angle
            rot_angle_list = np.arange(start, stop, step)
            fitness_list = []
            best_angles_list = []
            for (i, rot_angle) in enumerate(rot_angle_list):
                (fitness, best_angles) = self._angle_fitness(rot_angle, initial_R, ending_vec, starting_vec)
                fitness_list.append(fitness)
                best_angles_list.append(best_angles)
            #Find the best result
            best_index = np.argmin(fitness_list)
            best_rot_angle = rot_angle_list[best_index]
            best_angles = best_angles_list[best_index]
            return (best_rot_angle, best_angles)


        def optimize_c_code(start, stop, step):
            """Routine to optimize by brute force"""
            #Go through every angle
            rot_angle_list = np.arange(start, stop, step)
            (best_rot_angle, best_angles) = self._angle_fitness_brute(rot_angle_list, initial_R, ending_vec, starting_vec)
            return (best_rot_angle, best_angles)

        args = (initial_R, ending_vec, starting_vec)

        if search_method:
            #--- scipy optimize ----

            # Get a starting point by brute force 
            step = np.deg2rad(2)
            (best_rot_angle, best_angles) = optimize_c_code(-2.2*pi, pi*2.2, step)

            # And optimize with that
            if False:
                x0 = best_rot_angle
                res = scipy.optimize.fminbound(self._angle_fitness, 0, 2*pi, args, xtol=4e-3, disp=0, maxfun=100, full_output=0)
                best_rot_angle = res
            else:
                x0 = np.array([ best_rot_angle ])
                res = scipy.optimize.fmin(self._angle_fitness_python, x0, args, xtol=4e-3, ftol=1e-2, disp=0, maxiter=100)
                best_rot_angle = res.reshape( (1) )[0] #avoid error with 0-dimension array

            #Call the same function to get the best angles too
            (fitness, best_angles) = self._angle_fitness_python(best_rot_angle, *args)

        else:
            #--- semi-brute optimization routine ----
            #for optimize_func in [optimize, optimize_c_code]:
            step = np.deg2rad(2)
            # (best_rot_angle, best_angles) = optimize_c_code(-0.2*pi, pi*2.2, step)
            (best_rot_angle, best_angles) = optimize_c_code(-1.2*pi, pi*1.2, step)
            for x in xrange(4):
                newstep = step/10
                (best_rot_angle, best_angles) = optimize_c_code(best_rot_angle-step, best_rot_angle+step, newstep)
                step = newstep

        #Optimized angles
        return best_angles




#===============================================================================================
#===============================================================================================
#===============================================================================================
class TestLimitedGoniometer(LimitedGoniometer):
    """Limited goniometer for testing."""


    #-------------------------------------------------------------------------
    def __init__(self, wavelength_control=False):
        """Constructor"""
        #Init the base class
        LimitedGoniometer.__init__(self, wavelength_control)

        #Some info about the goniometer
        self.name = "Testing Limited Goniometer"
        self.description = "Limited goniometer for testing."

        #Make the angle info object
        self.gonio_angles = [
            AngleInfo('Phi', friendly_range=[-57, 0], random_range=[-1.0, 0.0]),
            AngleInfo('Chi', friendly_range=[0, 57], random_range=[0, 1]),
            AngleInfo('Omega', friendly_range=[172, 229], random_range=[3, 4]),
            ]

#===============================================================================================
#===============================================================================================
#===============================================================================================

#===============================================================================================
class TOPAZCryoGoniometer(LimitedGoniometer):
    """TOPAZ cryogoniometer that only has omega rotational freedom with chi fixed at 0.0"""


    #Chi is 0
    chi = Float(0, label="Fixed Chi angle (deg)", desc="the fixed Chi angle that the goniometer has, in degrees.")

    view = View(Item('name'), Item('description'),
                Item('wavelength_control'),
                Item('wavelength_bandwidth', visible_when="wavelength_control"),       
                Item('wavelength_minimum', visible_when="wavelength_control"),       
                Item('wavelength_maximum', visible_when="wavelength_control"),
                Item('chi'), Item('angles_desc', style='readonly'))

    #-------------------------------------------------------------------------
    def __init__(self, wavelength_control=False):
        """Constructor"""
        #Init the base class
        LimitedGoniometer.__init__(self, wavelength_control)

        #Some info about the goniometer
        self.name = "TOPAZ CryoGoniometer"
        self.description = "TOPAZ Cryogoniometer with one degree of freedom (omega), with chi fixed at 0 degrees."

        self.chi = +0

        #Make the angle info object
        self.gonio_angles = [
            AngleInfo('Omega'),
            ]

    #-------------------------------------------------------------------------
    def __eq__(self, other):
        """Return True if the contents of self are equal to other."""
        return LimitedGoniometer.__eq__(self,other) and \
            (np.deg2rad(self.chi) == other.chi)

    #-------------------------------------------------------------------------
    def get_fitness_function_c_code(self):
        #C code for the fitness of phi,chi, omega.
        args = []
        for i in xrange(1):
            # Each angle
            for j in xrange(2):
                args.append(self.gonio_angles[i].random_range[j])
        # Last argument is the fixed chi value.
        args.append( np.deg2rad(self.chi) )
        args = tuple(args)

        s = """
        FLOAT fitness_function(FLOAT phi, FLOAT chi, FLOAT omega)
        {
            FLOAT phi_min = %f;
            FLOAT phi_max = %f;

            FLOAT chi_mid = %f;
            FLOAT phi_mid = (phi_min + phi_max) / 2;

            FLOAT fitness = absolute(chi - chi_mid)*10.0 + absolute(phi - phi_mid)/10.0;

            // Big penalties for being out of the range
            if (phi < phi_min) fitness += (phi_min - phi) * 1.0;
            if (phi > phi_max) fitness += (phi - phi_max) * 1.0;

            return fitness;
        }""" % (args)
        return s

    #-------------------------------------------------------------------------------
    def get_phi_chi_omega(self, angles):
        """Given a list of angles (which may have more or less angles depending on goniometer type),
        return the equivalent (phi, chi, omega) in radians."""
        phi = angles[0]
        chi = np.deg2rad(self.chi)
        omega = 0
        return (phi, chi, omega)

    #-------------------------------------------------------------------------------
    def make_q_rot_matrix(self, angles):
        """Generate the necessary rotation matrix for use in the getq method.
        The q rotation matrix corresponds to the opposite (negative) angles that
        are the sample rotation angles.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        #For other instruments, this method may be different.
        (phi, chi, omega) = self.get_phi_chi_omega(angles)

        #In Q space, detector coverage rotates OPPOSITE to what the real space rotation is.
        #Because that is where the detectors and incident beam go, AS SEEN BY THE SAMPLE.
        #So wee need to invert the sample orientation matrix to find the one that will apply to the Q vector.
        return numpy_utils.opposite_rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------------
    def make_sample_rot_matrix(self, angles):
        """Generate the sample rotation matrix, from the given sample orientation angles.
        Unlike make_q_rot_matrix(), the direct angles are used here.
        This matrix will be used to calculate the scattering angle of specific reflections.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        (phi, chi, omega) = self.get_phi_chi_omega(angles)
        return numpy_utils.rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------
    def calculate_angles_to_rotate_vector(self, *args, **kwargs):
        """Calculate a set of sample orientation angles that rotate a single vector.
        TRY to return a sample orientation that is achievable by the goniometer.

        Parameters:
            see  LimitedGoniometer.calculate_angles_to_rotate_vector()

        Return:
            best_angles: list of the 2 angles found. None if invalid inputs were given
        """
        #The parent class does the work
        best_angles = LimitedGoniometer.calculate_angles_to_rotate_vector(self, *args, **kwargs)

        if best_angles is None:
            return None
        else:
            (phi, chi, omega) = best_angles
            
            if not np.abs(chi - np.deg2rad(self.chi)) < 0.5/57:
                # Have some tolerance (1 deg) in chi to help find anything. 
                return None
            else:
                #Okay, we found a decent chi
                return [omega]

#===============================================================================================
#===============================================================================================
#===============================================================================================
class SNAPLimitedGoniometer(LimitedGoniometer):
    """SNAP goniometer that only has omega rotational freedom with chi fixed at 0.0"""


    #Chi is 0
    chi = Float(0, label="Fixed Chi angle (deg)", desc="the fixed Chi angle that the goniometer has, in degrees.")

    view = View(Item('name'), 
                Item('description'),
                Item('wavelength_control'),
                Item('wavelength_bandwidth', visible_when="wavelength_control"),        
                Item('wavelength_minimum', visible_when="wavelength_control"),        
                Item('wavelength_maximum', visible_when="wavelength_control"),
                Item('chi'), Item('angles_desc', style='readonly'))

    #-------------------------------------------------------------------------
    def __init__(self, wavelength_control=False):
        """Constructor"""
        #Init the base class
        LimitedGoniometer.__init__(self, wavelength_control)

        #Some info about the goniometer
        self.name = "SNAP Goniometer"
        self.description = "SNAP goniometer with one degree of freedom (Phi), with chi fixed at 0 degrees."

        self.chi = +0

        #Make the angle info object
        self.gonio_angles = [
            AngleInfo('Phi'),
            ]

    #-------------------------------------------------------------------------
    def __eq__(self, other):
        """Return True if the contents of self are equal to other."""
        return LimitedGoniometer.__eq__(self,other) and \
            (np.deg2rad(self.chi) == other.chi)

    #-------------------------------------------------------------------------
    def get_fitness_function_c_code(self):
        #C code for the fitness of phi,chi, omega.
        args = []
        for i in xrange(1):
            # Each angle
            for j in xrange(2):
                args.append(self.gonio_angles[i].random_range[j])
        # Last argument is the fixed chi value.
        args.append( np.deg2rad(self.chi) )
        args = tuple(args)

        s = """
        FLOAT fitness_function(FLOAT phi, FLOAT chi, FLOAT omega)
        {
            FLOAT phi_min = %f;
            FLOAT phi_max = %f;

            FLOAT chi_mid = %f;
            FLOAT phi_mid = (phi_min + phi_max) / 2;

            FLOAT fitness = absolute(chi - chi_mid)*10.0 + absolute(phi - phi_mid)/10.0;

            // Big penalties for being out of the range
            if (phi < phi_min) fitness += (phi_min - phi) * 1.0;
            if (phi > phi_max) fitness += (phi - phi_max) * 1.0;

            return fitness;
        }""" % (args)
        return s


    #-------------------------------------------------------------------------------
    def get_phi_chi_omega(self, angles):
        """Given a list of angles (which may have more or less angles depending on goniometer type),
        return the equivalent (phi, chi, omega) in radians."""
        phi = angles[0]
        chi = np.deg2rad(self.chi)
        omega = 0
        return (phi, chi, omega)

    #-------------------------------------------------------------------------------
    def make_q_rot_matrix(self, angles):
        """Generate the necessary rotation matrix for use in the getq method.
        The q rotation matrix corresponds to the opposite (negative) angles that
        are the sample rotation angles.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        #For other instruments, this method may be different.
        (phi, chi, omega) = self.get_phi_chi_omega(angles)

        #In Q space, detector coverage rotates OPPOSITE to what the real space rotation is.
        #Because that is where the detectors and incident beam go, AS SEEN BY THE SAMPLE.
        #So wee need to invert the sample orientation matrix to find the one that will apply to the Q vector.
        return numpy_utils.opposite_rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------------
    def make_sample_rot_matrix(self, angles):
        """Generate the sample rotation matrix, from the given sample orientation angles.
        Unlike make_q_rot_matrix(), the direct angles are used here.
        This matrix will be used to calculate the scattering angle of specific reflections.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        (phi, chi, omega) = self.get_phi_chi_omega(angles)
        return numpy_utils.rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------
    def calculate_angles_to_rotate_vector(self, *args, **kwargs):
        """Calculate a set of sample orientation angles that rotate a single vector.
        TRY to return a sample orientation that is achievable by the goniometer.

        Parameters:
            see  LimitedGoniometer.calculate_angles_to_rotate_vector()

        Return:
            best_angles: list of the 2 angles found. None if invalid inputs were given
        """
        #The parent class does the work
        best_angles = LimitedGoniometer.calculate_angles_to_rotate_vector(self, *args, **kwargs)

        if best_angles is None:
            return None
        else:
            (phi, chi, omega) = best_angles
            
            if not np.abs(chi - np.deg2rad(self.chi)) < 0.5/57:
                # Have some tolerance (1 deg) in chi to help find anything. 
                return None
            else:
                #Okay, we found a decent chi
                return [omega]



#===============================================================================================
#===============================================================================================
#===============================================================================================
class MandiGoniometer(LimitedGoniometer):
    """Goniometer for MANDI instrument. Totally free in phi rotation, no freedom otherwise"""

    #Chi is 130
    chi = Float(+130.0, label="Fixed Chi angle (deg)", desc="the fixed Chi angle that the goniometer has, in degrees.")
    #Omega is 90
    omega = Float(+90.0, label="Fixed Omega angle (deg)", desc="the fixed Omega angle that the goniometer has, in degree.")

    view = View(Item('name'), Item('description'),
                Item('wavelength_control'),
                Item('wavelength_bandwidth', visible_when="wavelength_control"),        Item('wavelength_minimum', visible_when="wavelength_control"),        Item('wavelength_maximum', visible_when="wavelength_control"),
                Item('chi'),Item('omega'), Item('angles_desc', style='readonly'))

    #-------------------------------------------------------------------------
    def __init__(self, wavelength_control=False):
        """Constructor"""
        #Init the base class
        LimitedGoniometer.__init__(self, wavelength_control)

        #Some info about the goniometer
        self.name = "Mandi goniometer"
        self.description = "Mandi goniometer with one degree of freedom (phi), with chi fixed at 130 degrees and omega at 90 degrees."

        #Chi is +130 degrees 
        self.chi = +130.0
        chi = np.deg2rad(self.chi)
        a= np.deg2rad(self.chi)
        #Omega is 90 degrees
        self.omega = +90.0

        #Make the angle info object
        self.gonio_angles = [
            AngleInfo('Phi', friendly_range=[0, 360], random_range=[0.0, np.deg2rad(360)])
            ]

    #-------------------------------------------------------------------------
    def __eq__(self, other):
        """Return True if the contents of self are equal to other."""
        return LimitedGoniometer.__eq__(self,other) and \
            (np.deg2rad(self.chi) == other.chi) and \
            (np.deg2rad(self.omega) == other.omega)

    #-------------------------------------------------------------------------
    def get_fitness_function_c_code(self):
        """C code for the fitness of phi,chi, omega.
        Fitness is always good since the goniometer has no limits"""
        s = """FLOAT fitness_function(FLOAT phi, FLOAT chi, FLOAT omega)
        {
            FLOAT fitness = absolute(phi);
            return fitness;
        }
        """ 
        return s


    #-------------------------------------------------------------------------------
    def get_phi_chi_omega(self, angles):
        """Given a list of angles (which may have more or less angles depending on goniometer type),
        return the equivalent (phi, chi, omega) in radians."""
        (phi) = angles[0]
        chi = np.deg2rad(self.chi)
        omega = np.deg2rad(self.omega)
        return (phi, chi, omega)

    #-------------------------------------------------------------------------------
    def make_q_rot_matrix(self, angles):
        """Generate the necessary rotation matrix for use in the getq method.
        The q rotation matrix corresponds to the opposite (negative) angles that
        are the sample rotation angles.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        #For other instruments, this method may be different.
        (phi, chi, omega) = self.get_phi_chi_omega(angles)

        #In Q space, detector coverage rotates OPPOSITE to what the real space rotation is.
        #Because that is where the detectors and incident beam go, AS SEEN BY THE SAMPLE.

        #So wee need to invert the sample orientation matrix to find the one that will apply to the Q vector.
        return numpy_utils.opposite_rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------------
    def make_sample_rot_matrix(self, angles):
        """Generate the sample rotation matrix, from the given sample orientation angles.
        Unlike make_q_rot_matrix(), the direct angles are used here.
        This matrix will be used to calculate the scattering angle of specific reflections.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        (phi, chi, omega) = self.get_phi_chi_omega(angles)
        return numpy_utils.rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------
    def calculate_angles_to_rotate_vector(self, *args, **kwargs):
        """Calculate a set of sample orientation angles that rotate a single vector.
        TRY to return a sample orientation that is achievable by the goniometer.

        Parameters:
            see  LimitedGoniometer.calculate_angles_to_rotate_vector()

        Return:
            best_angles: list of the 2 angles found. None if invalid inputs were given
        """
        #The parent class does the work
        best_angles = LimitedGoniometer.calculate_angles_to_rotate_vector(self, *args, **kwargs)

        if best_angles is None:
            return None
        else:
            (phi, chi, omega) = best_angles
            #Chi needs to be 130 degrees! So we take it out

            if not np.abs(chi - np.deg2rad(self.chi)) < 0.1/57 and  not np.abs(omega - np.deg2rad(self.omega)) < 0.1/57:
                #Chi is not within +-0.1 degree of the fixed chi value degrees!
                #Omega is not within +-0.1 degree of the fixed chi value degrees!
                #print "Warning! Found angles", np.rad2deg(best_angles), " where chi is more than 1 degree off of fixed value."
                return None
            else:
                #Okay, we found a decent phi
                return [phi]

#===============================================================================================
#===============================================================================================
#===============================================================================================
class MandiVaryOmegaGoniometer(LimitedGoniometer):
    """Ambient goniometer with two degrees of freedom (phi and omega), with chi fixed at +45 degrees."""

    #Chi is +130 degrees 
    chi = Float(+130.0, label="Fixed Chi angle (deg)", desc="the fixed Chi angle that the goniometer has, in degrees.")

    view = View(Item('name'), Item('description'),
                Item('wavelength_control'),
                Item('wavelength_bandwidth', visible_when="wavelength_control"),        Item('wavelength_minimum', visible_when="wavelength_control"),        Item('wavelength_maximum', visible_when="wavelength_control"),
                Item('chi'), Item('angles_desc', style='readonly'))

    #-------------------------------------------------------------------------
    def __init__(self, wavelength_control=False):
        """Constructor"""
        #Init the base class
        LimitedGoniometer.__init__(self, wavelength_control)

        #Some info about the goniometer
        self.name = "MandiVaryOmega Goniometer"
        self.description = "Ambient goniometer with two degrees of freedom (phi and omega), with chi fixed at +135 degrees."

        #Chi is +130 degrees 
        self.chi = +130.0

        #Make the angle info object
        self.gonio_angles = [
            AngleInfo('Phi', friendly_range=[0, 360], random_range=[0.0, np.deg2rad(360)]),
            AngleInfo('Omega', friendly_range=[0, 360], random_range=[0.0, np.deg2rad(360)])
            ]

    #-------------------------------------------------------------------------
    def __eq__(self, other):
        """Return True if the contents of self are equal to other."""
        return LimitedGoniometer.__eq__(self,other) and \
            (np.deg2rad(self.chi) == other.chi)

    #-------------------------------------------------------------------------
    def get_fitness_function_c_code(self):
        #C code for the fitness of phi,chi, omega.
        args = []
        for i in xrange(2):
            for j in xrange(2):
                args.append(self.gonio_angles[i].random_range[j])
        # Last argument is the fixed chi value.
        args.append( np.deg2rad(self.chi) )
        args = tuple(args)

        s = """
        FLOAT fitness_function(FLOAT phi, FLOAT chi, FLOAT omega)
        {
            FLOAT phi_min = %f;
            FLOAT phi_max = %f;
            FLOAT omega_min = %f;
            FLOAT omega_max = %f;

            FLOAT phi_mid = (phi_min + phi_max) / 2;
            FLOAT chi_mid = %f;
            FLOAT omega_mid = (omega_min + omega_max) / 2;

            FLOAT fitness = absolute(chi - chi_mid)*10.0 + absolute(omega - omega_mid)/10.0 + absolute(phi - phi_mid)/10.0;

            // Big penalties for being out of the range
            if (phi < phi_min) fitness += (phi_min - phi) * 1.0;
            if (phi > phi_max) fitness += (phi - phi_max) * 1.0;
            if (omega < omega_min) fitness += (omega_min - omega) * 1.0;
            if (omega > omega_max) fitness += (omega - omega_max) * 1.0;

            return fitness;
        }""" % (args)
        return s


    #-------------------------------------------------------------------------------
    def get_phi_chi_omega(self, angles):
        """Given a list of angles (which may have more or less angles depending on goniometer type),
        return the equivalent (phi, chi, omega) in radians."""
        (phi, omega) = angles[0:2]
        chi = np.deg2rad(self.chi)
        return (phi, chi, omega)

    #-------------------------------------------------------------------------------
    def make_q_rot_matrix(self, angles):
        """Generate the necessary rotation matrix for use in the getq method.
        The q rotation matrix corresponds to the opposite (negative) angles that
        are the sample rotation angles.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        #For other instruments, this method may be different.
        (phi, omega) = angles[0:2]
        chi = np.deg2rad(self.chi)

        #In Q space, detector coverage rotates OPPOSITE to what the real space rotation is.
        #Because that is where the detectors and incident beam go, AS SEEN BY THE SAMPLE.

        #So wee need to invert the sample orientation matrix to find the one that will apply to the Q vector.
        return numpy_utils.opposite_rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------------
    def make_sample_rot_matrix(self, angles):
        """Generate the sample rotation matrix, from the given sample orientation angles.
        Unlike make_q_rot_matrix(), the direct angles are used here.
        This matrix will be used to calculate the scattering angle of specific reflections.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        (phi, omega) = angles[0:2]
        chi = np.deg2rad(self.chi)
        return numpy_utils.rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------
    def calculate_angles_to_rotate_vector(self, *args, **kwargs):
        """Calculate a set of sample orientation angles that rotate a single vector.
        TRY to return a sample orientation that is achievable by the goniometer.

        Parameters:
            see  LimitedGoniometer.calculate_angles_to_rotate_vector()

        Return:
            best_angles: list of the 2 angles found. None if invalid inputs were given
        """
        #The parent class does the work
        best_angles = LimitedGoniometer.calculate_angles_to_rotate_vector(self, *args, **kwargs)

        if best_angles is None:
            return None
        else:
            (phi, chi, omega) = best_angles
            #Chi needs to be 45 degrees! So we take it out

            if not np.abs(chi - np.deg2rad(self.chi)) < 0.1/57:
                #Chi is not within +-0.1 degree of the fixed chi value degrees!
                #print "Warning! Found angles", np.rad2deg(best_angles), " where chi is more than 1 degree off of fixed value."
                return None
            else:
                #Okay, we found a decent chi
                return [phi, omega]


#===============================================================================================
#===============================================================================================
#===============================================================================================
class ImagineGoniometer(LimitedGoniometer):
    """Goniometer for IMAGINE instrument. Totally free in phi rotation, no freedom otherwise"""

    #Chi is 0 
    chi = Float(0, label="Fixed Chi angle (deg)", desc="the fixed Chi angle that the goniometer has, in degrees.")

    view = View(Item('name'), Item('description'),
                Item('wavelength_control'),
                Item('wavelength_bandwidth', visible_when="wavelength_control"),        Item('wavelength_minimum', visible_when="wavelength_control"),        Item('wavelength_maximum', visible_when="wavelength_control"),
                Item('chi'), Item('angles_desc', style='readonly'))

    #-------------------------------------------------------------------------
    def __init__(self, wavelength_control=False):
        """Constructor"""
        #Init the base class
        LimitedGoniometer.__init__(self, wavelength_control)

        #Some info about the goniometer
        self.name = "IMAGINE goniometer"
        self.description = "IMAGINE goniometer with one degree of freedom (phi), with chi fixed at 0 degrees."

        #Chi is +135 degrees as of October 2010.
        self.chi = 0

        #Make the angle info object
        self.gonio_angles = [
            AngleInfo('Phi')
            ]

    #-------------------------------------------------------------------------
    def __eq__(self, other):
        """Return True if the contents of self are equal to other."""
        return LimitedGoniometer.__eq__(self,other) and \
            (np.deg2rad(self.chi) == other.chi)

    #-------------------------------------------------------------------------
    def get_fitness_function_c_code(self):
        """C code for the fitness of phi,chi, omega.
        Fitness is always good since the goniometer has no limits"""
        s = """FLOAT fitness_function(FLOAT phi, FLOAT chi, FLOAT omega)
        {
            FLOAT fitness = absolute(phi);
            return fitness;
        }
        """ 
        return s


    #-------------------------------------------------------------------------------
    def get_phi_chi_omega(self, angles):
        """Given a list of angles (which may have more or less angles depending on goniometer type),
        return the equivalent (phi, chi, omega) in radians."""
        (phi) = angles[0]
        chi = np.deg2rad(self.chi)
        omega = 0
        return (phi, chi, omega)

    #-------------------------------------------------------------------------------
    def make_q_rot_matrix(self, angles):
        """Generate the necessary rotation matrix for use in the getq method.
        The q rotation matrix corresponds to the opposite (negative) angles that
        are the sample rotation angles.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        #For other instruments, this method may be different.
        (phi, chi, omega) = self.get_phi_chi_omega(angles)

        #In Q space, detector coverage rotates OPPOSITE to what the real space rotation is.
        #Because that is where the detectors and incident beam go, AS SEEN BY THE SAMPLE.

        #So wee need to invert the sample orientation matrix to find the one that will apply to the Q vector.
        return numpy_utils.opposite_rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------------
    def make_sample_rot_matrix(self, angles):
        """Generate the sample rotation matrix, from the given sample orientation angles.
        Unlike make_q_rot_matrix(), the direct angles are used here.
        This matrix will be used to calculate the scattering angle of specific reflections.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        (phi, chi, omega) = self.get_phi_chi_omega(angles)
        return numpy_utils.rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------
    def calculate_angles_to_rotate_vector(self, *args, **kwargs):
        """Calculate a set of sample orientation angles that rotate a single vector.
        TRY to return a sample orientation that is achievable by the goniometer.

        Parameters:
            see  LimitedGoniometer.calculate_angles_to_rotate_vector()

        Return:
            best_angles: list of the 2 angles found. None if invalid inputs were given
        """
        #The parent class does the work
        best_angles = LimitedGoniometer.calculate_angles_to_rotate_vector(self, *args, **kwargs)

        if best_angles is None:
            return None
        else:
            (phi, chi, omega) = best_angles
            #Chi needs to be 45 degrees! So we take it out

            if not np.abs(chi - np.deg2rad(self.chi)) < 0.1/57:
                #Chi is not within +-0.1 degree of the fixed chi value degrees!
                #print "Warning! Found angles", np.rad2deg(best_angles), " where chi is more than 1 degree off of fixed value."
                return None
            else:
                #Okay, we found a decent chi
                return [phi, omega]



#===============================================================================================
#===============================================================================================
#===============================================================================================
class ImagineMiniKappaGoniometer(LimitedGoniometer):
    """Goniometer for IMAGINE instrument. MiniKappa"""

    #Alpha is 24 degrees
    alpha = Float(24.0, label="Fixed Alpha angle (deg)", desc="the fixed Chi angle that the goniometer has, in degrees.")

    view = View(Item('name'), 
                Item('description'),
                Item('wavelength_control'),
                Item('wavelength_bandwidth', visible_when="wavelength_control"),        
                Item('wavelength_minimum', visible_when="wavelength_control"),        
                Item('wavelength_maximum', visible_when="wavelength_control"),
                Item('alpha'), 
                Item('angles_desc', style='readonly'))

    #-------------------------------------------------------------------------
    def __init__(self, wavelength_control=False):
        """Constructor"""
        #Init the base class
        LimitedGoniometer.__init__(self, wavelength_control)

        #Some info about the goniometer
        self.name = "IMAGINE mini-kappa goniometer"
        self.description = "IMAGINE mini-kappa goniometer with three degrees of freedom (phi,kappa,omega), with alpha fixed at 24 degrees."

        #Alpha is 24 degrees
        self.alpha = 24.0

        #Make the angle info object
        #Not sure if these limits match IMAGINE's mini-kappa
        self.gonio_angles = [
            AngleInfo('Phi', friendly_range=[75, 235], random_range=[np.deg2rad(75), np.deg2rad(235)]),
            AngleInfo('Chi', friendly_range=[0, 48], random_range=[0, np.deg2rad(48)]),
            AngleInfo('Omega', friendly_range=[225, 489], random_range=[np.deg2rad(225), np.deg2rad(489)]),
            ]

    #-------------------------------------------------------------------------
    def get_fitness_function_c_code(self):
        #C code for the fitness of phi, kappa, omega.
        args = []
        for i in xrange(3):
            for j in xrange(2):
                args.append(self.gonio_angles[i].random_range[j])
        args = tuple(args)
            
        s = """
        FLOAT fitness_function(FLOAT phi, FLOAT chi, FLOAT omega)
        {
            FLOAT phi_min = %f;
            FLOAT phi_max = %f;
            FLOAT chi_min = %f;
            FLOAT chi_max = %f;
            FLOAT omega_min = %f;
            FLOAT omega_max = %f;

            FLOAT phi_mid = (phi_min + phi_max) / 2;
            FLOAT chi_mid = (chi_min + chi_max) / 2;
            FLOAT omega_mid = (omega_min + omega_max) / 2;

            FLOAT fitness =  absolute(chi - chi_mid) + absolute(omega - omega_mid) + absolute(phi - phi_mid);

            // Big penalties for being out of the range
            if (phi < phi_min) fitness += (phi_min - phi) * 1.0;
            if (phi > phi_max) fitness += (phi - phi_max) * 1.0;
            if (chi < chi_min) fitness += (chi_min - chi) * 1.0;
            if (chi > chi_max) fitness += (chi - chi_max) * 1.0;
            if (omega < omega_min) fitness += (omega_min - omega) * 1.0;
            if (omega > omega_max) fitness += (omega - omega_max) * 1.0;

            // if (phi < phi_min || phi > phi_max) fitness += 10;
            // if (chi < chi_min || chi > chi_max) fitness += 10;
            // if (omega < omega_min || omega > omega_max) fitness += 10;

            return fitness;
        }""" % (args)
        return s

    #-------------------------------------------------------------------------------
    def get_phi_chi_omega(self, angles):
        """Given a list of angles (which may have more or less angles depending on goniometer type),
        return the equivalent (phi, kappa, omega) in radians."""
        (phi) = angles[0]
        (chi) = angles[1]
        (omega) = angles[2]
        return (phi, chi, omega)

    #-------------------------------------------------------------------------------
    def make_q_rot_matrix(self, angles):
        """Generate the necessary rotation matrix for use in the getq method.
        The q rotation matrix corresponds to the opposite (negative) angles that
        are the sample rotation angles.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        #For other instruments, this method may be different.
        (phi, chi, omega) = self.get_phi_chi_omega(angles)

        #In Q space, detector coverage rotates OPPOSITE to what the real space rotation is.
        #Because that is where the detectors and incident beam go, AS SEEN BY THE SAMPLE.

        #So wee need to invert the sample orientation matrix to find the one that will apply to the Q vector.
        return numpy_utils.opposite_rotation_matrix(phi, chi, omega)

    #-------------------------------------------------------------------------------
    def make_sample_rot_matrix(self, angles):
        """Generate the sample rotation matrix, from the given sample orientation angles.
        Unlike make_q_rot_matrix(), the direct angles are used here.
        This matrix will be used to calculate the scattering angle of specific reflections.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        (phi, chi, omega) = self.get_phi_chi_omega(angles)
        return numpy_utils.rotation_matrix(phi, chi, omega)
    

    #-------------------------------------------------------------------------
    def calculate_angles_to_rotate_vector(self, *args, **kwargs):
        """Calculate a set of sample orientation angles that rotate a single vector.
        TRY to return a sample orientation that is achievable by the goniometer.

        Parameters:
            see  LimitedGoniometer.calculate_angles_to_rotate_vector()

        Return:
            best_angles: list of the 2 angles found. None if invalid inputs were given
        """
        #The parent class does the work
        best_angles = LimitedGoniometer.calculate_angles_to_rotate_vector(self, *args, **kwargs)

        if best_angles is None:
            return None
        else:
            (phi, chi, omega) = best_angles
            # Check that all angles are within allowable ranges, or return none
            if  self.gonio_angles[0].is_angle_valid(phi) and \
                self.gonio_angles[1].is_angle_valid(chi) and \
                self.gonio_angles[2].is_angle_valid(omega):
                    return best_angles
            else:
                    return None

    #===============================================================================================
    def csv_make_header(self, fileobj, title, comment=""):
        """Make the header text of the motor positions CSV file. This is general for universal
        goniometers, but can be subclassed by specific ones.

        Parameters:
        -----------
            fileobj: an already open, writable file object.
        """
        #Line of header info
        fileobj.write(csv_line( ['Comment', 'phi', 'kappa', 'omega', 'Wait For', 'Value'] ) )


    #===============================================================================================
    def csv_add_position(self, fileobj, angle_values, count_for, count_value, comment):
        """Add a line to an existing CSV file containing the motor positions, etc. for
        that part of the experiment.

        Parameters:
            angles: list of angles in the sample orientation.
            count_for, count_value: stopping criterion
        """
        #Calculate if its allowed
        (allowed, reason) = self.are_angles_allowed(angle_values, return_reason=True)
        #Convert from internal to DAS units.
        
        phi, chi, omega = angle_values
        alpha = np.deg2rad(self.alpha)
        phi, alpha, kappa, omega = numpy_utils.kappa_from_euler(phi, chi, omega, alpha=alpha)
        angles = [phi, kappa, omega]
        
        das_angles = [self.angles[i].internal_to_das(angles[i]) for i in xrange(len(self.angles))]
        stopping_criterion = count_for
        if stopping_criterion == "runtime":
            stopping_criterion = "seconds"
        if not allowed:
            #Can't reach this position
            fileobj.write("#"" ----- ERROR! This sample orientation could not be achieved with the goniometer, because of '%s'. THE FOLLOWING LINE HAS BEEN COMMENTED OUT ------ ""\n" % reason )
            fileobj.write('#' + csv_line( comment, das_angles + [stopping_criterion, count_value] ) )
        else:
            #They are okay
            fileobj.write(csv_line( [comment] + das_angles + [stopping_criterion, count_value] ) )

#===============================================================================================
#===============================================================================================
#===============================================================================================
class TopazAmbientGoniometer(LimitedGoniometer):
    """Ambient goniometer with two degrees of freedom (phi and omega), with chi fixed at +45 degrees."""

    #Chi is +135 degrees as of October 2010.
    chi = Float(+135.0, label="Fixed Chi angle (deg)", desc="the fixed Chi angle that the goniometer has, in degrees.")

    view = View(Item('name'), Item('description'),
                Item('wavelength_control'),
                Item('wavelength_bandwidth', visible_when="wavelength_control"),        Item('wavelength_minimum', visible_when="wavelength_control"),        Item('wavelength_maximum', visible_when="wavelength_control"),
                Item('chi'), Item('angles_desc', style='readonly'))

    #-------------------------------------------------------------------------
    def __init__(self, wavelength_control=False):
        """Constructor"""
        #Init the base class
        LimitedGoniometer.__init__(self, wavelength_control)

        #Some info about the goniometer
        self.name = "TOPAZ Ambient Goniometer"
        self.description = "Ambient goniometer with two degrees of freedom (phi and omega), with chi fixed at +135 degrees."

        #Chi is +135 degrees as of October 2010.
        self.chi = +135.0

        #Make the angle info object  (after EPICS upgrade names changed)
        self.gonio_angles = [
            AngleInfo('BL12:Mot:goniokm:phi'),
            AngleInfo('BL12:Mot:goniokm:omega'),
            ]

    #-------------------------------------------------------------------------
    def __eq__(self, other):
        """Return True if the contents of self are equal to other."""
        return LimitedGoniometer.__eq__(self,other) and \
            (np.deg2rad(self.chi) == other.chi)

    #-------------------------------------------------------------------------
    def get_fitness_function_c_code(self):
        #C code for the fitness of phi,chi, omega.
        args = []
        for i in xrange(2):
            for j in xrange(2):
                args.append(self.gonio_angles[i].random_range[j])
        # Last argument is the fixed chi value.
        args.append( np.deg2rad(self.chi) )
        args = tuple(args)

        s = """
        FLOAT fitness_function(FLOAT phi, FLOAT chi, FLOAT omega)
        {
            FLOAT phi_min = %f;
            FLOAT phi_max = %f;
            FLOAT omega_min = %f;
            FLOAT omega_max = %f;

            FLOAT phi_mid = (phi_min + phi_max) / 2;
            FLOAT chi_mid = %f;
            FLOAT omega_mid = (omega_min + omega_max) / 2;

            FLOAT fitness = absolute(chi - chi_mid)*10.0 + absolute(omega - omega_mid)/10.0 + absolute(phi - phi_mid)/10.0;

            // Big penalties for being out of the range
            if (phi < phi_min) fitness += (phi_min - phi) * 1.0;
            if (phi > phi_max) fitness += (phi - phi_max) * 1.0;
            if (omega < omega_min) fitness += (omega_min - omega) * 1.0;
            if (omega > omega_max) fitness += (omega - omega_max) * 1.0;

            return fitness;
        } """ % (args)
        return s


    #-------------------------------------------------------------------------------
    def get_phi_chi_omega(self, angles):
        """Given a list of angles (which may have more or less angles depending on goniometer type),
        return the equivalent (phi, chi, omega) in radians."""
        (phi, omega) = angles[0:2]
        chi = np.deg2rad(self.chi)
        return (phi, chi, omega)

    #-------------------------------------------------------------------------------
    def make_q_rot_matrix(self, angles):
        """Generate the necessary rotation matrix for use in the getq method.
        The q rotation matrix corresponds to the opposite (negative) angles that
        are the sample rotation angles.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        #For other instruments, this method may be different.
        (phi, omega) = angles[0:2]
        chi = np.deg2rad(self.chi)

        #In Q space, detector coverage rotates OPPOSITE to what the real space rotation is.
        #Because that is where the detectors and incident beam go, AS SEEN BY THE SAMPLE.

        #So wee need to invert the sample orientation matrix to find the one that will apply to the Q vector.
        return numpy_utils.opposite_rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------------
    def make_sample_rot_matrix(self, angles):
        """Generate the sample rotation matrix, from the given sample orientation angles.
        Unlike make_q_rot_matrix(), the direct angles are used here.
        This matrix will be used to calculate the scattering angle of specific reflections.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        (phi, omega) = angles[0:2]
        chi = np.deg2rad(self.chi)
        return numpy_utils.rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------
    def calculate_angles_to_rotate_vector(self, *args, **kwargs):
        """Calculate a set of sample orientation angles that rotate a single vector.
        TRY to return a sample orientation that is achievable by the goniometer.

        Parameters:
            see  LimitedGoniometer.calculate_angles_to_rotate_vector()

        Return:
            best_angles: list of the 2 angles found. None if invalid inputs were given
        """
        #The parent class does the work
        best_angles = LimitedGoniometer.calculate_angles_to_rotate_vector(self, *args, **kwargs)

        if best_angles is None:
            return None
        else:
            (phi, chi, omega) = best_angles
            #Chi needs to be 45 degrees! So we take it out

            if not np.abs(chi - np.deg2rad(self.chi)) < 0.1/57:
                #Chi is not within +-0.1 degree of the fixed chi value degrees!
                #print "Warning! Found angles", np.rad2deg(best_angles), " where chi is more than 1 degree off of fixed value."
                return None
            else:
                #Okay, we found a decent chi
                return [phi, omega]

    #===============================================================================================
    def csv_make_header(self, fileobj, title, comment=""):
        """Make the header text of the motor positions CSV file. This is general for universal
        goniometers, but can be subclassed by specific ones.

        Parameters:
        -----------
            fileobj: an already open, writable file object.
        """
        #Line of header info
        
        fileobj.write(csv_line( ['Notes'] + [x.name for x in self.angles] + ['Wait For/n', 'Value'] ) )


    #===============================================================================================
    def csv_add_position(self, fileobj, angle_values, count_for, count_value, comment):
        """Add a line to an existing CSV file containing the motor positions, etc. for
        that part of the experiment.

        Parameters:
            angles: list of angles in the sample orientation.
            count_for, count_value: stopping criterion
        """
        #Calculate if its allowed
        (allowed, reason) = self.are_angles_allowed(angle_values, return_reason=True)
        #Convert from internal to DAS units.
        das_angles = [self.angles[i].internal_to_das(angle_values[i]) for i in xrange(len(self.angles))]
        stopping_criterion = count_for
        if stopping_criterion == "runtime":
            stopping_criterion = "seconds"
        elif stopping_criterion == "pcharge":
            stopping_criterion = "BL12:Det:PCharge:C"
        if not allowed:
            #Can't reach this position
            fileobj.write("#"" ----- ERROR! This sample orientation could not be achieved with the goniometer, because of '%s'. THE FOLLOWING LINE HAS BEEN COMMENTED OUT ------ ""\n" % reason )
            fileobj.write('#' + csv_line( [comment] + das_angles + [stopping_criterion, count_value] ) )
        else:
            #They are okay
            fileobj.write(csv_line( [comment] + das_angles + [stopping_criterion, count_value] ) )




#===============================================================================================
#===============================================================================================
#===============================================================================================
class TopazInHouseGoniometer(LimitedGoniometer):
    """This is the "in-house" TOPAZ goniometer, composed of 3 legs movable in X/Y that
    control the tilt of a sample plate.
    """

    #Some info about the goniometer
    name = "TOPAZ In-House Goniometer"
    description = "3-legged goniometer for TOPAZ, which allows for sample cooling. Has full phi rotation freedom, but is limited in chi and omega due to the travel range of the legs."


    #---------------------------- GONIOMETER DEFAULT VALUES -----------------------
    #Default inhouse goniometer object file.
    #Positions are all in mm.

    #Height of the SAMPLE plate above the sample position/beam height, when all legs
    # are at the zero position.
    sample_plate_height = 3.141 * 25.4
    #Therefore the SAMPLE plate's Y = +sample_plate_height (again, at the zero
    #position).

    #Length of each mounting leg, mm
    leg_length = 4.196 * 25.4

    #Height (y-position) of the FIXED plate. This is fixed relative to the
    #sample (and beam) position. Origin is the SAMPLE position.
    fixed_plate_height = sin(pi/3)*leg_length + sample_plate_height

    #The 3 mounting points on the SAMPLE plate form an equilateral triangle,
    # with sides this long:
    mounting_side_length = 2.815*25.4

    #Vector describing the position of the sample relative to the expected, in mm
    #   sample position, when the sample rotation motor is zero-ed (see below).
    #   The sample sits on a pin, gonio.sample_plate_height BELOW the sample plate.
    relative_sample_position = column([0, 0, 0])

    #Angle of the small rotational motor on the sample plate. This will be some
    #   kind of stepper motor. Will use the convention that when motor_phi = 0,
    #   the motor is pointed towards positive z (downstream).
    sample_motor_phi = 0

    #The rotational motor is less precise, so it will take care of the larger movements.
    #   Since it is a stepper, it can only do angles in steps.
    #   phi rotations will be handled up to this precision:
    sample_motor_step_size = pi / 360  #1 degree.

    #This is the offset in rotation that the sample motor has relative to the zero we use
    #(where motor_phi = 0 points downstream). This is used to correct for any
    #misalignment.
    sample_motor_zero_offset = 0

    #MAXIMUM amount of travel (in both directions from center) that each leg has on the fixed plate, in mm
    travel = 15

    #Leg safe zone is a 3D array giving the XZ coordinates that each legs are allowed to travel to
    #   in the fixed plate (set to True), false elsewhere.
    #Coords are (leg#, X, Z)
    leg_safe_zone = None
    #These axes define the position (X and Z) for a given position in the 2D array.
    leg_safe_xaxis = None
    leg_safe_zaxis = None
    #...using this resolution, in mm
    leg_safe_resolution = 0.5

    #Fault (out-of-range) error in each lef
    leg_fault = array([False, False, False])

    #Special objects stored only for the purposes of animating the data
    _plot_sample_plate = None
    _plot_fixed_plate = None
    _plot_legs = [None, None, None]
    _plot_pin = None
    _plot_motor = None

    #--- For the allowable angles map ----
    #Range that we will explore for phi, +- around 0
    phi_range = sample_motor_step_size / 2
    #List of angles to consider
    phi_list = None
    chi_list = None
    omega_list = None

    #Limit to angular res. (for the allowed coverage map).
    ANGULAR_RESOLUTION_MIN = 0.001

    #And this is the map of allowed angles, a 3D array
    allowed = None


    #===============================================================================================
    def __init__(self, *args, **kwargs):
        """Constructor."""
        LimitedGoniometer.__init__(self, *args, **kwargs)
        
        #Make the arrays with the XY limits of leg movement
        self.calculate_leg_xy_limits(visualize=False)
        #Make the angle info object
        self.gonio_angles = [AngleInfo('Phi'),
                    AngleInfo('Chi', random_range=[np.deg2rad(-12), np.deg2rad(+12)]),
                    AngleInfo('Omega', random_range=[np.deg2rad(-12), np.deg2rad(+12)])]



    #===============================================================================================
    def are_angles_allowed(self, angles, return_reason=False):
        """Calculate whether the given angles can be reached with the goniometer.

        Parameters:
            angles: list of angles of the goniometer (phi, chi, omega), in radians.
            return_reason: bool; return the reason string as a tuple

        Returns:
            allowed: True if the angle provided can be reached.
            [optionally] reason, a string. Return value is then a tuple
        """
        if len(angles) != 3:
            raise ValueError("In-house goniometer expects 3 angles (phi, chi, omega).")

        def check_these(phi, chi, omega):
            """Check a set of angles"""
            #Calculate the leg position from these angles
            self.getplatepos(phi, chi, omega)
            #Make sure the legs are not in fault
            self.check_limits()
            return (not any(self.leg_fault))

        (phi, chi, omega) = angles
        allowed = check_these(phi, chi, omega)
        if not allowed:
            #This set of goniometer angles is always equivalent, resulting
            #   in the same rotation matrix!
            allowed = check_these(phi-pi, -chi, omega-pi)

        if return_reason:
            reason = ""
            if not allowed: reason = "Limit to leg positions"
            return (allowed, reason)
        else:
            return allowed



    #===============================================================================================
    def getplatepos(self, phi=0, chi=0, omega=0):
        """Calculate the XYZ coordinates of the sample and fixed plates for the
        inhouse goniometer for the specified sample rotation. Also calculates the sample
        motor rotation.

        Parameters:
        -----------
           self: goniometer object containing all parameters.
           phi, chi, omega: angles (in radians) corresponding to ISAW standard
                           rotations.
        
           Movements are performed in this (virtual) order:
               1. Rotation of the sample phi motor.
               2. Translation to bring sample into beam position.
               3. Movement (tilting) of the sample plate by moving the legs.
        
        Fields set in the goniometer object:
            angles: array with all 3 angles.
            sample_plate: position of the legs on the sample (movable) plate.
            fixed_plate: position of the legs on the fixed plate.
            sample_motor_phi: rotation of the sample (phi) motor.
            pin, motor and sample_middle: vectors showing positions of the sample,
                motor and middle of the sample plate."""

        #Save the specified angles in the structure
        angles = np.array([phi, chi, omega]);

        #We divvy up the phi rotation between the plate and the sample motor.
        #We round to the nearest multiple of the sample motor step size.
        self.sample_motor_phi = round(phi / self.sample_motor_step_size) * self.sample_motor_step_size
        #And the remainder is handled by the sample plate position.
        sample_plate_phi = phi - self.sample_motor_phi

        #This calculates the rotation matrix for the sample PLATE only.
        rot_M_plate = rotation_matrix(sample_plate_phi, chi, omega)

        #And this is the rotation matrix for the sample motor only
        rot_M_motor = rotation_matrix(self.sample_motor_phi, 0, 0)


        #X,Y,Z translation vector (in mm) to perform BEFORE moving the sample plate.
        #To calculate these, we use the relative_sample_position vector.
        translate_v = -self.relative_sample_position
        #But we have to correct for the sample motor phi rotation by rotating the translation
        #vector as well.
        translate_v = np.dot(rot_M_motor, translate_v)
        

        #------------------ SAMPLE PLATE ----------------------
        #3 vectors representing the position of the mounting points on the plate,
        #when it is horizontal and with the sample at 0
        #Remember, the plate is in the X-Z plane.

        #distance between center of plate and each mounting point.
        d = self.mounting_side_length / (2 * np.cos(pi / 6))
        #Distance to the edge on the other side
        d2 = np.sin(pi / 6) * d

        #Vectors representing the sample plate at the "zero" position.
        sample_plate_zero = np.column_stack(([self.mounting_side_length / 2, self.sample_plate_height, d2],
            [-self.mounting_side_length / 2, self.sample_plate_height, d2],
            [0, self.sample_plate_height, -d]))

        #------------------ OTHER USEFUL POINTS ----------------------
        #Vector representing the position of the middle of the sample plate.
        sample_middle = column([0, self.sample_plate_height, 0])

        #Make a vector representing the position of the sample at the end of the
        #pin.
        pin = self.relative_sample_position

        #Make vector to represent the sample motor orientation (at zero)
        self.motor_vector_length = 20
        motor = column([0, self.sample_plate_height, self.motor_vector_length])


        #------------------ APPLY TRANSFORMS ----------------------
        #For the sample plate: we do not apply the motor_phi rotation.
        
        #Do a translation of the position - we are moving the entire sample plate
        #   This places the sample in the 0,0,0 position.
        sample_plate = get_translated_vectors(sample_plate_zero, translate_v)

        #Now do a rotation (phi,chi,omega)
        sample_plate = dot(rot_M_plate, sample_plate)

        #The pin rotates with the motor, then translates, then then rotates with the
        #sample plate.
        pin = dot(rot_M_motor, pin)
        pin = get_translated_vectors(pin, translate_v)
        pin = dot(rot_M_plate, pin)

        #Motor vector = same as pin.
        motor = dot(rot_M_motor, motor)
        motor = get_translated_vectors(motor, translate_v)
        motor = dot(rot_M_plate, motor)

        #Same for the sample_middle vector
        sample_middle = dot(rot_M_motor, sample_middle)
        sample_middle = get_translated_vectors(sample_middle, translate_v)
        sample_middle = dot(rot_M_plate, sample_middle)

        #Sample plate coordinates are:
        #i.e. x_A2, y_A2, x_B2, etc. (as written in Janik's notebook)

        #We want to find the positions of the other ends of the legs on the fixed
        #plate, x_A1, etc.
        fixed_plate = np.copy(sample_plate)

        #Legs A and B are fixed in their orientation along Z, and C along X, so we
        #know the Z_A1, Z_B1 and X_C1 positions on the FIXED plate are the same as
        #on the SAMPLE plate.

        #We also know the height of all these points, y = fixed_plate_height.
        fixed_plate[COORD_Y, :] = self.fixed_plate_height
      
        #This leaves x_A1, x_B1, and z_C1 to find.

        #Angle between the x direction and the (A1 to A2) vector formed by leg A
        theta_A = np.arcsin((sample_plate[COORD_Y, MOUNT_A] - self.fixed_plate_height) / self.leg_length)
        if theta_A > -pi / 2:
            #Force theta_A to be ~-120 degrees
            theta_A = -pi - theta_A
        

        #Angle between the x direction and the B1 to B2) vector formed by leg B
        theta_B = np.arcsin((sample_plate[COORD_Y, MOUNT_B] - self.fixed_plate_height) / self.leg_length)

        #We can easily calculate the x position from these
        x_A1 = sample_plate[COORD_X, MOUNT_A] - self.leg_length * cos(theta_A)
        x_B1 = sample_plate[COORD_X, MOUNT_B] - self.leg_length * cos(theta_B)

        fixed_plate[COORD_X, MOUNT_A] = x_A1
        fixed_plate[COORD_X, MOUNT_B] = x_B1


        #Finally we find the position of Leg C
        phi_C = np.arcsin((sample_plate[COORD_Y, MOUNT_C] - self.fixed_plate_height) / self.leg_length)
        if phi_C < -pi / 2:
            #Force phi_C to be ~-60 degrees
            phi_C = 2*pi + phi_C

        #Now we calc. the Z position of leg C on the fixed plate.
        z_C1 = sample_plate[COORD_Z, MOUNT_C] - self.leg_length * cos(phi_C)
        fixed_plate[COORD_Z, MOUNT_C] = z_C1


        #Assign these plate position in the goniometer object, which is returned
        self.sample_plate = sample_plate
        self.fixed_plate = fixed_plate
        self.sample_plate_zero = sample_plate_zero

        #Also return the pin and motor vectors
        self.pin = pin
        self.motor = motor
        self.sample_middle = sample_middle



    #===============================================================================================
    def calculate_leg_xy_limits(self, visualize=False):
        """Create a arrays showing the limits limits to the legs' positions.
        This will eventually be replaced with experimentally-determined values.

            visualize: Plot the resulting map."""
           
        #Find the fixed plate position at the "0" point
        gonio_zero = copy.copy(self)
        gonio_zero.relative_sample_position = column([0.0, 0.0, 0.0]) #Tell the sample to be centered well.
        gonio_zero.getplatepos(0.0, 0.0, 0.0)
        fixed_plate_zero = np.copy(gonio_zero.fixed_plate)
        #This defines the center of the following matrices
        self.fixed_plate_zero = fixed_plate_zero
        
        #Now we generate a matrix of allowed positions around those points.
        self.leg_safe_xaxis = np.arange(-self.travel, self.travel, self.leg_safe_resolution)
        self.leg_safe_zaxis = np.copy(self.leg_safe_xaxis)

        #Create the "safe zone" array, initialized to False
        self.leg_safe_zone = np.zeros( (3, self.leg_safe_xaxis.size, self.leg_safe_zaxis.size), dtype=bool ) 

        #Now make a reasonable approximation
        real_travel_x = 12.5
        real_travel_z = real_travel_x
        for leg in range(3):
            for i_x in range(self.leg_safe_xaxis.size):
                x = self.leg_safe_xaxis[i_x]
                if abs(x)<real_travel_x:
                    for i_z in range(self.leg_safe_zaxis.size):
                        z = self.leg_safe_zaxis[i_z]
                        if abs(z)<real_travel_z:
                            self.leg_safe_zone[leg, i_x, i_z] = True
#        #Upper left corner of leg A (0)
#        center = int(len(self.leg_safe_xaxis)/2)
#        self.leg_safe_zone[0, :, :] = False
#        self.leg_safe_zone[0, :center, :center] = True
#        self.leg_safe_zone[1, :, :] = False
#        self.leg_safe_zone[1, center:, 0:center] = True
#        self.leg_safe_zone[2, :, :center] = False


        if visualize:
            pylab.figure(0)
            pylab.hold(True)
            for leg in range(3):
                pylab.pcolor(self.leg_safe_xaxis+fixed_plate_zero[COORD_X, leg], self.leg_safe_zaxis+fixed_plate_zero[COORD_Z, leg], self.leg_safe_zone[leg, :, :].transpose())
            pylab.xlabel("x")
            pylab.ylabel("z")
            pylab.title("Allowable XZ leg positions for the 3 legs.")
            pylab.draw()
            pylab.axis('equal')
            #pylab.show()

    #===============================================================================================
    def check_limits(self):
        """Check that the leg positions in self.fixed_plate are within the limits previously
        calculated using self.calculate_leg_xy_limits().
        
            This sets the self.leg_fault array of booleans to True for each leg outside of its allowed
        range."""

        #Find the relative position of each leg vs. its "zero" position
        relpos = self.fixed_plate - self.fixed_plate_zero

        for leg in range(3):
            #Check that the leg is within allowable "safe zone"
            #Use the position of the leg (relative to 0) to find the index in the "safe zone" matrix
            i_x = nearest_index(self.leg_safe_xaxis, relpos[COORD_X, leg])
            i_z = nearest_index(self.leg_safe_zaxis, relpos[COORD_Z, leg])
            #Look up in the safe zone.
            self.leg_fault[leg] = (not self.leg_safe_zone[leg, i_x, i_z])

            if (not all(np.isreal(self.fixed_plate[:, leg]))) or any(np.isnan(self.fixed_plate[:, leg])):
                #A complex or NaN value = the angle found for the leg was invalid, meaning that the
                #leg would have to be longer to reach the desired position.
                self.leg_fault[leg] = True


    #===============================================================================================
    def calculate_allowable_angles(self, angular_resolution=0.01, visualize=0, optimized=False):
        """Calculate a map of allowed phi, chi, omega angles.
        
            angular_resolution: angle to use as the grid size (rad)
            visualize: set to 1 or 2 to plot the results in 2D or 3D.
            optimized: use simple optimization to reduce calculation time. Assumes that the 
                shape of allowed angles is contiguous, which could be false for a custom-made
                disallowed area!
        """
        #Could be optimized more, if required.
        
        if visualize:
            pylab.ion() #Turns interaction on (?)
            print "Calculation started"

        if angular_resolution < self.ANGULAR_RESOLUTION_MIN:
            warnings.warn("Requested too small angular resolution. Using %s" % self.ANGULAR_RESOLUTION_MIN)
            angular_resolution = self.ANGULAR_RESOLUTION_MIN

        #We do not need to look at more than the motor step size in phi, since the sample rotational
        #   motor does the rest.
        self.phi_range = 0.99 * (self.sample_motor_step_size / 2)
        #Arbitrary max ranges here. TODO: User setting?
        self.chi_range = pi/8
        self.omega_range = pi/6

        #Make the lists
        self.phi_list = array([-self.phi_range, 0, self.phi_range])
        self.phi_list = np.append(np.arange(-self.phi_range, self.phi_range, angular_resolution), (self.phi_range) )
        self.chi_list = np.arange(-self.chi_range, self.chi_range, angular_resolution)
        self.omega_list = np.arange(-self.omega_range, self.omega_range, angular_resolution)

        #Create the allowed angle map, all false
        self.allowed = np.zeros( (self.phi_list.size, self.chi_list.size, self.omega_list.size), dtype=np.bool )

        #Make sure the limits are calculated
        self.calculate_leg_xy_limits()


        #Loop through all the angles
        for i_phi in range(self.phi_list.size):
            phi = self.phi_list[i_phi]
            print 'Calculating phi = %s' % phi
            was_anything_in_line = False
            for i_chi in range(self.chi_list.size):
                chi = self.chi_list[i_chi]
                was_allowed = False
                for i_omega in range(self.omega_list.size):
                    omega = self.omega_list[i_omega]

                    #Calcualte the leg position from these angles
                    self.getplatepos(phi, chi, omega)
                    #Make sure the legs are not in fault
                    self.check_limits()
                    allowed = not any(self.leg_fault)
                    #Save it
                    self.allowed[i_phi, i_chi, i_omega] = allowed
                    if optimized:
                        #Check that we have reached an unallowed section, to speed calculation
                        if allowed: was_allowed = True
                        if was_allowed and (not allowed):
                            break #All other omega values in this chi will be false. Exit to the next line
                    #(end of omega loop)

                if optimized:
                    #Is there anything allowed in this chi line?
                    anything_in_line = any(self.allowed[i_phi, i_chi, :])
                    #Once we reach a completely disallowed line, all others after will be too
                    if anything_in_line: was_anything_in_line = True
                    if was_anything_in_line and (not anything_in_line):
                        break #we don't need to look at the remaining lines.
                #(end of chi loop)
            #(end of phi loop)

        print "Done!"
                    
        if visualize==1:
            #2D Animated plot
            for i_phi in range(self.phi_list.size):
                phi = self.phi_list[i_phi]
                #Here we are still in the phi loop
                if visualize:
                    pylab.figure(0)
                    pylab.imshow(self.allowed[i_phi, :, :], interpolation='nearest', extent=(-self.omega_range, self.omega_range, -self.chi_range, self.chi_range))
                    pylab.xlabel("Omega")
                    pylab.ylabel("Chi")
                    pylab.title("Allowable angles for phi=%s" % phi)
                    pylab.draw()
                    time.sleep(0.2)
                    pylab.draw()
            time.sleep(1)
            pylab.ion()
            

        if visualize==2:
            #Single 3D plot of allowed angles.
            mlab.clf() #Clear figure
            mlab.contour3d(self.allowed.astype(float), contours=[0.5])
            mlab.xlabel("Phi")
            mlab.ylabel("Chi")
            mlab.zlabel("Omega")
            mlab.axes()
            mlab.show()

        #Done calculating
        return None


    #===============================================================================================
    def plot_goniometer(self, first_plot=False):
        """Plot the goniometer geometry to a window.

            first_plot: Set to true when doing the first plot of a sequence/animation.
        """

        #Middle of the 3 points on the sample plate.
        calculated_sample_middle = column(np.mean(self.sample_plate, 1))

        #Check if there is any fault to the legs.
        self.check_limits()

        #-------- Some tests --------
        difference = self.sample_middle - calculated_sample_middle
        error = vector_length(difference)
        if abs(error) > 0.01:
            warning('The middle of the sample plate\'s 3 points does not match the middle calculated with sample motor rotation!\nError is %f' % difference)

        if vector_length(self.pin) > 0.01:
            warning(mcat([mstring('The calculated sample position is not zero after rotation/translation!'), 10, mstring('This means that the calculation failed to keep the sample centered, somehow.'), 10, mstring('Sample is at: '), num2str(self.pin.cT)]))

        error = vector_length(self.motor - self.sample_middle) - self.motor_vector_length
        if abs(error) > 0.01:
            warning(mcat([mstring('The calculated length of the motor vector after rotation does not match the expected length!'), mstring('Error is '), num2str(error)]))

        #---- Start plotting -----
        if first_plot:
            mlab.clf()
            draw_cartesian_axes(20, np.array([0, 150, 0]))
            #Plot The "zero" position sample plate
            plot3dpatch(self.sample_plate_zero, (0, 1, 1), 0.5)
            #Show the range of movement of the legs on the fixed plate
            plot3dpoints(self.fixed_plate_zero, scale_factor=7)
            text3d(self.fixed_plate_zero[:, MOUNT_A], 'A', color=(0,0,0))
            text3d(self.fixed_plate_zero[:, MOUNT_B], 'B', color=(0,0,0))
            text3d(self.fixed_plate_zero[:, MOUNT_C], 'C', color=(0,0,0))

#            #The limits
#            for leg in range(3):
#                extent = [self.leg_safe_xaxis[0]+self.fixed_plate_zero[COORD_X, leg],
#                        self.leg_safe_xaxis[-1]+self.fixed_plate_zero[COORD_X, leg],
#                        self.fixed_plate_zero[COORD_Y, leg],
#                        self.fixed_plate_zero[COORD_Y, leg],
#                        self.leg_safe_xaxis[0]+self.fixed_plate_zero[COORD_Z, leg],
#                        self.leg_safe_xaxis[-1]+self.fixed_plate_zero[COORD_Z, leg]]
#                print extent
#                imagedata = np.copy(self.leg_safe_zone[leg, :, :])
#                print imagedata
#                mlab.imshow(imagedata, extent=extent)
                
            #Clear all the plotobjects
            self._plot_sample_plate = None
            self._plot_fixed_plate = None
            self._plot_legs = [None, None, None]
            self._plot_pin = None
            self._plot_motor = None
            self._plot_title = None
            self._plot_leg_points = None


        #For any plots after
        #Draw the rotated, translated sample plate
        self._plot_sample_plate = plot3dpatch(self.sample_plate, (1,0,0), 1, plotobj=self._plot_sample_plate)

        #Draw the fixed plate position
        self._plot_fixed_plate = plot3dpatch(self.fixed_plate, (0,1,0), 0.5, plotobj=self._plot_fixed_plate)
        self._plot_leg_points = plot3dpoints(self.fixed_plate, color=(0,1,0), plotobj=self._plot_leg_points)


        #Add arrows for legs
        for leg in range(3):
            color = (0,0,0)
#            if self.leg_fault[leg]: color = (1,0,0)
            self._plot_legs[leg] =  arrow(self.sample_plate[:, leg], self.fixed_plate[:, leg], color=color, plotobj=self._plot_legs[leg])

        #Arrow for the sample pin
        self._plot_pin = arrow(self.sample_middle, self.pin, plotobj=self._plot_pin)
        #Arrow for motor direction
        color = (0, 0.5, 0)
        self._plot_motor = arrow(self.sample_middle, self.motor, color=color, plotobj=self._plot_motor)
        #Point in the middle
#        plot3dpoints(calculated_sample_middle)

        #Add title
        if self._plot_title is None:
            self._plot_title = mlab.title(title_string, size=0.2)
        else:
            #This doesn't work :(
            self._plot_title.set(title_string)

    #===============================================================================================
    def animate(self, phi_list, chi_list, omega_list, delay=0.02):
        """Calculate and display the position of the goniometer. You need to have called
        plot_goniometer(first_plot=True) at least once before.
        
            phi_list,chi_list,omega_list: lists of the angles (in radians) to use.
            delay: keep the image on for this many seconds.
        """
        for phi in phi_list:
            for chi in chi_list:
                for omega in omega_list:
                    self.getplatepos(phi, chi, omega)
                    self.plot_goniometer()
                    time.sleep(delay)
                    pass

#
#    #===============================================================================================
#    def find_angles(self, angle_guess=None, adjust_position=True):
#        """Get the phi, chi, omega sample orientation angles that correspond to the known
#        positions of the legs on the fixed plate of the goniometer.
#        The goniometer object should have its leg coordinates already specified via:
#            self.fixed_plate
#        adjust_position: The code will try to optimize for the sample position. Otherwise, you
#            should set the self.relative_sample_position vector to the known sample offset.
#        """
#        #TODO: Complete this code, if ever necessary! Does not converge reliably yet.
#
#        #Build an initial guess array containing either the angles and position, or just angles
#        if angle_guess is None: angle_guess = np.array([0,0,0])
#        if adjust_position:
#            guess = np.concatenate((vector(angle_guess), vector(self.relative_sample_position)))
#        else:
#            guess = angle_guess
#
#        #Now we create a goniometer object that has the same geometry by copying it.
#        #It will be used by the error function to recalculate geometry.
#        gonio_test = copy.deepcopy(self)
#
#
#        def calculate_error(angles):
#            """Calculates the error between the supplied
#            leg positions and the expected leg positions for the given angles (and sample position vector).
#            get_angles calls to minimize this function."""
#            if adjust_position:
#                gonio_test.relative_sample_position = column(angles[3:6])
#            gonio_test.getplatepos(angles[0], angles[1], angles[2])
#            #Difference in the positions of the legs
#            difference = gonio_test.fixed_plate - self.fixed_plate
##            print difference
#            error = np.sum(np.sum(np.vdot(difference, difference)))
#            print error
#            return error
#
#
##        guess = ndarray([10])
#        print 'Fmin starting...'
#        xopt = scipy.optimize.fmin(calculate_error, guess)
#        print xopt

    #===============================================================================================
    def get_leg_coordinates(self):
        """Return the leg coordinates as a list of:
            [phi, LegA_X,LegA_Y,LegB_X,LegB_Y,LegC_X,LegC_Y]
                ... where phi is the sample phi motor position in radians
                ... where X is the X direction on the plate, also
                    corresponding to +X of real space.
                ... where Y is the other direction on the plate,
                    corresponding to +Z of real space (instrument coordinates).
        """
        #Start with the phi motor
        output = [self.sample_motor_phi]
        #fixed_plate_offset
        fixed_plate_offset = self.fixed_plate - self.fixed_plate_zero
        #Mounts A B C
        for mount in xrange(3):
            output.append(fixed_plate_offset[COORD_X, mount])
            #This is the Y value
            output.append(fixed_plate_offset[COORD_Z, mount])
        #List has 6 elements
        return output

    
    #===============================================================================================
    def csv_make_header(self, fileobj, title, comment=""):
        """Make the header text of the motor positions CSV file.

        Parameters:
        -----------
            fileobj: an already open, writable file object.
        """
        fileobj.write(csv_line( ["#Title:", title] ) )
        fileobj.write(csv_line( ["#Comment:", comment] ) )
        #Any other useful comment s trings?
        fileobj.write('#"First column is the sample phi motor rotation, in radians"\n' )
        fileobj.write('#"Next 6 columns are the XY leg positions in mm, relative to the central (neutral) position."\n' )
        fileobj.write('#"Next are 2 columns for the stopping criterion parameters."\n' )
        #Line of header info
        fileobj.write(csv_line( ['Phi', 'LegA_X', 'LegA_Y', 'LegB_X', 'LegB_Y', 'LegC_X', 'LegC_Y', 'CountFor', 'CountValue',  'Comment'] ) )


    #===============================================================================================
    def csv_add_position(self, fileobj, angles, count_for, count_value, comment):
        """Add a line to an existing CSV file.
        """
        #Blank line for pretty ness
        fileobj.write("#\n")
        #Make a comment line with the angles
        angle_comment = '#"Equivalent goniometer angles (phi, chi, omega) are: ' + string.join(["%.3f" % np.rad2deg(x) for x in angles], ", ") + ' degrees."\n'
        fileobj.write(angle_comment)
        #Calculate if its allowed
        (allowed, reason) = self.are_angles_allowed(angles, return_reason=True)
        #Get coordinates
        legs = self.get_leg_coordinates()
        if not allowed:
            #Can't reach this position
            fileobj.write("#"" ----- ERROR! This sample orientation could not be achieved with the goniometer, because of '%s'. THE FOLLOWING LINE HAS BEEN COMMENTED OUT ------ ""\n" % reason )
            fileobj.write('#' + csv_line( legs + [count_for, count_value, comment] ) )
        else:
            #They are okay
            fileobj.write(csv_line( legs + [count_for, count_value, comment] ) )




#===============================================================================================
#===============================================================================================
#===============================================================================================
class HB3AGoniometer(LimitedGoniometer):
    """Goniometer on the HFIR HB3A four-circle diffractometer.
    """

    #-------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        """Constructor"""
        #Init the base class
        LimitedGoniometer.__init__(self, wavelength_control=False)

        #Some info about the goniometer
        self.name = "HB3A Goniometer"
        self.description = "Goniometer for HFIR HB3A. Four degrees of freedom: phi, chi, omega, and detector."

        #Make the angle info object
        self.gonio_angles = [
            AngleInfo('Phi', friendly_range=[-180, 180]),
            AngleInfo('Chi', friendly_range=[-90, 90]),
            AngleInfo('Omega', friendly_range=[0, 50]),
            AngleInfo('Detector', friendly_range=[-1, 95]),
            ]


    #-------------------------------------------------------------------------
    def get_fitness_function_c_code(self):
        # C code for the fitness of phi,chi, omega.
        # We want omega to be close to its middle range of 25 degrees.
        # Phi is free
        return """
        FLOAT fitness_function(FLOAT phi, FLOAT chi, FLOAT omega)
        {
            double center = 3.14159*25.0/180.0;
            double omegadiff = omega - center;
            if (omegadiff < 0) omegadiff = -omegadiff;
             
            //if (omegadiff > center)
            // omegadiff = omegadiff + (omega-center) * 10.0;
            
            return absolute(chi) + omegadiff + absolute(phi)/1000.0;
            //return absolute(chi) + omegadiff + absolute(phi);
        }"""


    #-------------------------------------------------------------------------
    def calculate_angles_to_rotate_vector(self, *args, **kwargs):
        """Calculate a set of sample orientation angles that rotate a single vector.
        TRY to return a sample orientation that is achievable by the goniometer.

        Parameters:
            see  LimitedGoniometer.calculate_angles_to_rotate_vector()

        Return:
            best_angles: list of the 2 angles found. None if invalid inputs were given
        """
        #The parent class does the work
        best_angles = LimitedGoniometer.calculate_angles_to_rotate_vector(self, *args, **kwargs)

        if best_angles is None:
            return None
        else:
            (phi, chi, omega) = best_angles

            # Check that all angles are within allowable ranges, or return none
            if  self.gonio_angles[0].is_angle_valid(phi) and \
                self.gonio_angles[1].is_angle_valid(chi) and \
                self.gonio_angles[2].is_angle_valid(omega):
                    return best_angles
            else:
                    return None

#===============================================================================================
#===============================================================================================
#===============================================================================================
class CorelliGoniometer(LimitedGoniometer):
    """Goniometer on Corelli
    """

    #-------------------------------------------------------------------------
    def __init__(self, wavelength_control=False):
        """Constructor"""
        #Init the base class
        LimitedGoniometer.__init__(self, wavelength_control)

        #Some info about the goniometer
        self.name = "Corelli Goniometer"
        self.description = "Goniometer for Corelli. Two degrees of freedom. Phi is from -175 to 150, Chi is from 1 to 35."

        #Just set omega to 0..
        self.omega = +0.0

        #Make the angle info object
        self.gonio_angles = [
            AngleInfo('Phi', friendly_range=[-175, 150]),
            AngleInfo('Chi', friendly_range=[1, 35]),
            ]

    #-------------------------------------------------------------------------
    def __eq__(self, other):
        """Return True if the contents of self are equal to other."""
        return LimitedGoniometer.__eq__(self,other) and \
            (np.deg2rad(self.omega) == other.omega)

    #-------------------------------------------------------------------------
    def get_fitness_function_c_code(self):
        #C code for the fitness of phi,chi, omega.
        args = []
        for i in xrange(2):
            for j in xrange(2):
                args.append(self.gonio_angles[i].random_range[j])
        # Last argument is the fixed chi value.
        args.append( np.deg2rad(self.omega) )
        args = tuple(args)

        s = """
        FLOAT fitness_function(FLOAT phi, FLOAT chi, FLOAT omega)
        {
            FLOAT phi_min = %f;
            FLOAT phi_max = %f;
            FLOAT omega_min = %f;
            FLOAT omega_max = %f;

            FLOAT phi_mid = (phi_min + phi_max) / 2;
            FLOAT chi_mid = %f;
            FLOAT omega_mid = (omega_min + omega_max) / 2;

            FLOAT fitness = absolute(chi - chi_mid)*10.0 + absolute(omega - omega_mid)/10.0 + absolute(phi - phi_mid)/10.0;

            // Big penalties for being out of the range
            if (phi < phi_min) fitness += (phi_min - phi) * 1.0;
            if (phi > phi_max) fitness += (phi - phi_max) * 1.0;
            if (omega < omega_min) fitness += (omega_min - omega) * 1.0;
            if (omega > omega_max) fitness += (omega - omega_max) * 1.0;

            return fitness;
        }""" % (args)
        return s


    #-------------------------------------------------------------------------------
    def get_phi_chi_omega(self, angles):
        """Given a list of angles (which may have more or less angles depending on goniometer type),
        return the equivalent (phi, chi, omega) in radians."""
        (phi, chi) = angles[0:2]
        omega = np.deg2rad(self.omega)
        return (phi, chi, omega)

    #-------------------------------------------------------------------------------
    def make_q_rot_matrix(self, angles):
        """Generate the necessary rotation matrix for use in the getq method.
        The q rotation matrix corresponds to the opposite (negative) angles that
        are the sample rotation angles.
        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        #For other instruments, this method may be different.
        (phi, chi) = angles[0:2]
        omega = np.deg2rad(self.omega)

        #In Q space, detector coverage rotates OPPOSITE to what the real space rotation is.
        #Because that is where the detectors and incident beam go, AS SEEN BY THE SAMPLE.

        #So wee need to invert the sample orientation matrix to find the one that will apply to the Q vector.
        return numpy_utils.opposite_rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------------
    def make_sample_rot_matrix(self, angles):
        """Generate the sample rotation matrix, from the given sample orientation angles.
        Unlike make_q_rot_matrix(), the direct angles are used here.
        This matrix will be used to calculate the scattering angle of specific reflections.

        Parameters:
            angles: should be a list of angle values, in unfriendly units, that matches the
                # of angles of this goniometer.
        """
        (phi, chi) = angles[0:2]
        omega = np.deg2rad(self.omega)
        return numpy_utils.rotation_matrix(phi, chi, omega)


    #-------------------------------------------------------------------------
    def calculate_angles_to_rotate_vector(self, *args, **kwargs):
        """Calculate a set of sample orientation angles that rotate a single vector.
        TRY to return a sample orientation that is achievable by the goniometer.

        Parameters:
            see  LimitedGoniometer.calculate_angles_to_rotate_vector()

        Return:
            best_angles: list of the 2 angles found. None if invalid inputs were given
        """
        #The parent class does the work
        best_angles = LimitedGoniometer.calculate_angles_to_rotate_vector(self, *args, **kwargs)

        if best_angles is None:
            return None
        else:
            (phi, chi, omega) = best_angles
            #Chi needs to be 45 degrees! So we take it out

            if not np.abs(omega - np.deg2rad(self.omega)) < 0.1/57:
                #Chi is not within +-0.1 degree of the fixed chi value degrees!
                #print "Warning! Found angles", np.rad2deg(best_angles), " where chi is more than 1 degree off of fixed value."
                return None
            else:
                #Okay, we found a decent chi
                return [phi, chi]

#================================================================================================
#================================================================================================
#================================================================================================



#================================================================================================
def test_plotting_and_others():
    #Create a sample goniometer
    g = TopazInHouseGoniometer()
    #print g.are_angles_allowed(0,0,0)

    g.relative_sample_position = column([0, 0, 0.5])
    g.getplatepos(0,0.0,0.0)
    g.calculate_leg_xy_limits(visualize=False)

    g.getplatepos(np.deg2rad(20), np.deg2rad(7), np.deg2rad(10))
    return 
    #g.plot_goniometer(first_plot=True)
    #mlab.show()

##    g.calculate_allowable_angles(0.015, visualize=2)
#    g.are_angles_allowed(0,0,0)

    #Now do some animation
    if False:
        g.plot_goniometer(first_plot=True)
        phi = 0.2
        chi = 0.15
        omega = 0.15
        g.animate(np.arange(0, phi, 0.01), [0], [0])
        g.animate([phi], np.arange(0, chi, 0.01), [0])
        g.animate([phi], [chi], np.arange(0, omega, 0.01))

    #This will leave the window open for manipulation; halts exec. of program until window is closed.
    mlab.show()



def test_angle_finding():
    g = TopazInHouseGoniometer()
    g.calculate_leg_xy_limits(visualize=False)

    #These should give phi,chi,omega = -1,-1,0 (in degrees) or something close
    starting_vec = np.array([ 0.16222142,  0.16222142, -0.97332853])
    ending_vec = np.array([ 0.18198751,  0.15906953, -0.97034913])
    g.calculate_angles_to_rotate_vector(starting_vec, ending_vec, starting_angles=None)
    
    g = TopazAmbientGoniometer()
    starting_vec = np.array([  1.25663706e+00 , -7.69468277e-17,  -3.14159265e+00] )
    ending_vec = np.array([ 0.79397495,  1.34719908, -3.00056654])
    g.calculate_angles_to_rotate_vector(starting_vec, ending_vec, starting_angles=None)



#===============================================================================================
def sample_pin_position_range():
    """Ways to test the range of XYZ motion of the sample pin, with different
    leg movement limits."""
    #Create a sample goniometer
    g = TopazInHouseGoniometer()

    #Initialize the leg limits
    g.relative_sample_position = column([0.0, 0.0, 0.0])
    g.getplatepos(0.0, 0.0, 0.0)
    g.calculate_leg_xy_limits(visualize=True)

#    if True:
#        pylab.show()
#        return

    n = 17
    positions = np.linspace(-8, 8, n) #Range calculated in mm
    allowed = np.zeros( (n,n,n) )
    for (ix, x) in enumerate(positions):
        print "Calculating x", x
        for (iy, y) in enumerate(positions):
            for (iz, z) in enumerate(positions):
                #Set up
                g.relative_sample_position = column([x, y, z])
                allowed[ix,iy,iz] = g.are_angles_allowed([0., 0., 0.], return_reason=False)

    #Do a plot

    pylab.figure(1, figsize=[15,15])
    pylab.title("Allowable XZ sample positions")
    for (iy, y) in enumerate(positions):
        print "At y of", y, ", # of points = ", np.sum( allowed[:, iy,:])
        if iy < 16:
            pylab.subplot(4,4,iy+1)
            pylab.pcolor(positions, positions, allowed[:, iy, :].transpose(), norm=pylab.Normalize(0, 1))
            pylab.xlabel("x")
            pylab.ylabel("z")
            pylab.title("y = %.3f mm" % y)
            pylab.draw()
            pylab.axis('equal')
    pylab.show()
    #pylab.





#===============================================================================================
# Global list of all available goniometers
goniometers = []

def initialize_goniometers():
    """Function creates all possible goniometers."""
    goniometers.append( Goniometer() )
    goniometers.append( LimitedGoniometer() )
    goniometers.append( TestLimitedGoniometer() )
    goniometers.append( TopazInHouseGoniometer() )
    goniometers.append( TopazAmbientGoniometer() )
    goniometers.append( TOPAZCryoGoniometer() )
    goniometers.append( SNAPLimitedGoniometer() )
    goniometers.append( MandiGoniometer() )
    goniometers.append( MandiVaryOmegaGoniometer() )
    goniometers.append( ImagineGoniometer() )
    goniometers.append( ImagineMiniKappaGoniometer() )
    goniometers.append( CorelliGoniometer() )

def get_goniometers_names():
    """Returns a list of all available goniometer names."""
    return [gon.name for gon in goniometers]


import unittest

#==================================================================
class TestGoniometers(unittest.TestCase):
    """Unit test for the Goniometers"""
    def setUp(self):
        pass

    def test_constructors(self):
        g = Goniometer()
        g = TopazAmbientGoniometer()
        g = TopazInHouseGoniometer()

    def test_constructors_wavelength_control(self):
        g = Goniometer(wavelength_control=True)
        assert len(g.angles)==4, "3 angles and 1 wavelength"

    def test_inhouse(self):
        g = TopazInHouseGoniometer()
        assert g.are_angles_allowed([0,0,0])
        assert not g.are_angles_allowed([0,1.0,0])

    def test_hb3a(self):
        g = HB3AGoniometer()
        assert g.are_angles_allowed([0,0,0,0])

    def test_limited_angle_finding(self):
        g = LimitedGoniometer()
        starting_vec = np.array([1,2,3]);
        ending_vec = np.array([2,2,3]);
        g.calculate_angles_to_rotate_vector(starting_vec, ending_vec, [0, 0, 0])

    def test_TestLimited_angle_finding(self):
        g = TestLimitedGoniometer()
        starting_vec = np.array([1,2,3]);
        ending_vec = np.array([2,2,3]);
        g.calculate_angles_to_rotate_vector(starting_vec, ending_vec, [0, 0, 0])
        g.calculate_angles_to_rotate_vector(starting_vec, ending_vec, [0, 0, 0], search_method=1)

    def test_ambient_angle_finding(self):
        g = TopazAmbientGoniometer()
        starting_vec = np.array([1,2,3]);
        ending_vec = np.array([2,2,3]);
        g.calculate_angles_to_rotate_vector(starting_vec, ending_vec, [0, 0, 0])

#===============================================================================================
if __name__ == "__main__":
    unittest.main()
#    initialize_goniometers()
#    sample_pin_position_range()
