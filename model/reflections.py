"""Module defining the Reflection object.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import numpy as np

#--- Model Imports ---
from numpy_utils import column, vector_length

#==================================================================
#==================================================================
#==================================================================
class Reflection():
    """The Reflection class holds data relevant to a single diffraction
    peak.
    """

    #------------------------------------------------------------
    def __init__(self, hkl, q_vector):
        """Constructor.

        Parameters:
            hkl: tuple of (h,k,l), the indices of the reflection in the reciprocal lattice system.
                Normally will be integer, but float would work too.
            q_vector: vector containing the corresponding q-vector for this
                reflection.
        """
        #h,k,l coordinates of the reflection
        (self.h, self.k, self.l) = hkl
        #Save as tuple too
        self.hkl = hkl
        #And the pre-calculated q-vector
        self.q_vector = q_vector

        #List holds the measurements.
        #   value: a list of tuples.
        #       each tuple holds poscovid, det, h,v,wl
        self.measurements = list()

        #Divergence (half-width) in radians of the scattered beam
        self.divergence = 0.0

        #Is the reflection a 'primary' one, e.g. in the simplest half- or quarter-sphere
        self.is_primary = False

        #Link to the actual primary reflection, if this is a secondary one.
        self.primary = None

        #List of equivalent reflections (from symmetry); only set for primary reflections
        self.equivalent = []

        #This is a temp. value used in a calculation
        self.considered = False

        #self.q_vector = np.dot(reciprocal_lattice, column(self.hkl))


    #----------------------------------------------------
    def __str__(self):
        """Return an informal string representing the reflection."""
        s = "Reflection at hkl " + str(self.hkl) + "; q-vector " + str(self.q_vector.flatten()) + "\n"
        s += "    Measurements: " + str(self.measurements)
        return s

    #----------------------------------------------------
    def get_q_norm(self):
        """Return the norm of the q-vector corresponding to this hkl"""
        return vector_length(self.q_vector)

    #----------------------------------------------------
    def get_d_spacing(self):
        """Return the d-spacing corresponding to this hkl"""
        return 1 / (vector_length(self.q_vector) / (2*np.pi))

    #----------------------------------------------------
    def add_measurement(self, poscov_id, detector_num, horizontal, vertical, wavelength, distance):
        """Saves a measurement to the list.

        Parameters:
            poscov_id: id of the PositionCoverage object
            detector_num: number of the detector (0-based array).
            horizontal: horizontal position in the detector coordinates (in mm)
            vertical: vertical position in the detector coordinates (in mm)
            wavelength: wavelength detected (in Angstroms)
            distance: distance between sample and spot on detector
        """
        #Add the tuple of data to the list
        self.measurements.append( (poscov_id, detector_num, horizontal, vertical, wavelength, distance) )

    #----------------------------------------------------
    def times_measured(self, position_ids=None, add_equivalent_ones=False):
        """Return how many times this peak was measured succesfully.
        Parameters:
            position_ids: list of position IDs to consider.
                Set to None to use all of them.
            add_equivalent_ones: add up the times measured of all
                the equivalent reflections too.
        """
        if position_ids is None:
            if add_equivalent_ones:
                num = 0
                for refl in self.equivalent:
                    num += len(refl.measurements)
                return num
            else:
                return len(self.measurements)
        else:
            #Count the matching ones
            num = 0
            if add_equivalent_ones:
                for refl in self.equivalent:
                    for data in refl.measurements:
                        if data[0] in position_ids:
                            num += 1
            else:
                for data in self.measurements:
                    if data[0] in position_ids:
                        num += 1
            return num



#==================================================================
#==================================================================
#==================================================================
class ReflectionMeasurement():
    """The ReflectionMeasurement class is used to show data about
    a single measurement of a single reflection.
    It is not saved in the Reflection class (to save memory), but is
    re-generated when the GUI wants it.
    """

    #-------------------------------------------------------------------------------
    def __init__(self, refl, measurement_num, divergence_deg=0.0):
        """Create the object.

        Parameters:
            refl: Parent reflection object
            measurement_num: which entry in the list of refl.measurements?
            divergence_deg: half-width of the beam divergence in degreens
        """
        self.refl = refl
        if refl is None:
            self.measurement_num = -1
            #Extract the components of the measurement
            (self.poscov_id, self.detector_num, self.horizontal, self.vertical,
                self.wavelength, self.distance) \
                = (0, 0, 0, 0, 0, 0)
        else:
            self.measurement_num = measurement_num
            #Extract the components of the measurement
            (self.poscov_id, self.detector_num, self.horizontal, self.vertical,
                self.wavelength, self.distance) \
                = refl.measurements[measurement_num]
        
        #Now calculate the widths of the peak on the detector
        self.peak_width = self.calculate_peak_width(self.distance, divergence_deg)

#        (self.horizontal_delta, self.vertical_delta, self.wavelength_delta) = \
#            experiment.exp.calculate_peak_width(refl.hkl, delta_hkl)

    #-------------------------------------------------------------------------------
    def make_sample_orientation_string(self):
        """Return a friendly string of the sample orientation angles."""
        import instrument
        # @type poscov: PositionCoverage
        for poscov in instrument.inst.positions:
            if id(poscov)==self.poscov_id:
                return instrument.inst.make_angles_string( poscov.angles )
        return ""

    #-------------------------------------------------------------------------------
    def calculate_peak_width(self, distance, divergence_deg):
        """Calculate the width(radius) on a detector plate of a single peak, given a distance and a divergence.

        Parameters:
            distance: distance between sample and detector, usually in mm.
            divergence_deg: half-width of scattered beam divergence in degrees.

        Returns:
            width: half-width on the detector face, same units as distance
        """
        div = np.deg2rad(divergence_deg)
        multiplier = np.abs(np.tan(div))
        if multiplier > 10:
            return 10*distance
        else:
            return distance * multiplier
    


#================================================================================
#============================ UNIT TESTING ======================================
#================================================================================
import unittest

#==================================================================
class TestReflection(unittest.TestCase):
    """Unit test for the FlatDetector, checks what angles hit this detector"""
    #----------------------------------------------------
    def setUp(self):
        hkl = column([1.0,2,3])
        q_vector = 1.0/hkl
        self.ref = Reflection(hkl, q_vector)

    def test_constructor(self):
        """Reflection->Test the constructor"""
        ref = self.ref
        assert ref.h == 1
        assert ref.k == 2
        assert ref.l == 3
        assert np.all(ref.hkl == column([1,2,3]))
        assert np.allclose(ref.q_vector, column([1.0, 0.5, 1.0/3])), "q-vector was set to %s" % ref.q_vector
        assert isinstance(ref.measurements, list), "Made a list."
        assert len(ref.measurements) == 0, "No items in list."

#    def test_q_vector(self):
#        """Reflection->Q-vector calculation."""
#        rec = np.identity(3)
#        rec[1,0] = 1.0 #a_star_y = 1
#        ref = Reflection((1,2,3), rec)
#        assert np.all(ref.q_vector == column([1,3,3])), "q-vector was calculated"

    def test_add_measurement(self):
        "Reflection->add_measurement()"
        ref = self.ref
        ref.add_measurement(670, 5, 12.5, 32.3, 2.45, 400)
        assert len(ref.measurements) == 1, "One item in list."
        assert ref.measurements[0] == (670, 5, 12.5, 32.3, 2.45, 400), "Data tuple saved well."
        ref.add_measurement(223, 8, 13.5, 1.3, 3.45, 400)
        assert len(ref.measurements) == 2, "Two items in list."
        assert ref.measurements[1] == (223, 8, 13.5, 1.3, 3.45, 400), "Data tuple #2 saved well."
        assert ref.times_measured() == 2, "times_measured() works."
        


#==================================================================
if __name__ == "__main__":
    unittest.main()



