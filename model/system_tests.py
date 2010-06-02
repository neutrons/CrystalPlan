"""System tests (as unit tests) that test various aspects of Crystal Plan.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id: reflections.py 1226 2010-05-06 17:26:02Z 8oz $

#--- General Imports ---
import numpy as np

import crystals
from crystals import Crystal
import instrument
import goniometer
import experiment
from experiment import Experiment, ParamPositions


#================================================================================
#============================ UNIT TESTING ======================================
#================================================================================
import unittest

#==================================================================
class TestReflection(unittest.TestCase):
    """Unit test for the FlatDetector, checks what angles hit this detector"""
    #----------------------------------------------------
    def setUp(self):
        pass


    #----------------------------------------------------
    #----------------------------------------------------
    #The following reciprocal matrices were provided by Dennis Mikkelson, calculated by ISAW.
    def test_reciprocal_matrix1(self):
        c = Crystal("quartz")
        c.lattice_angles_deg = np.fromstring("90.0  90.0 120.0", sep=' ')
        c.lattice_lengths = np.fromstring("4.913  4.913  5.400", sep=' ')
        c.calculate_reciprocal()
        isaw_rec = np.fromstring("""   1.278890    0.000000    0.000000
   0.738367    1.476735   -0.000000
  -0.000000   -0.000000    1.163553""", sep=' ').reshape(3,3)
        assert np.allclose(isaw_rec, c.reciprocal_lattice), "Reciprocal lattice of %s matches ISAW lattice." % c.name

    def test_reciprocal_matrix2(self):
        c = Crystal("oxalic acid")
        c.lattice_angles_deg = np.fromstring("90.0 103.2  90.0", sep=' ')
        c.lattice_lengths = np.fromstring("6.094  3.601 11.915", sep=' ')
        c.calculate_reciprocal()
        isaw_rec = np.fromstring("""   1.031045    0.000000    0.000000
  -0.000000    1.744845    0.000000
   0.241829   -0.000000    0.54164""", sep=' ').reshape(3,3)
        assert np.allclose(isaw_rec, c.reciprocal_lattice), "Reciprocal lattice of %s matches ISAW lattice." % c.name

    def test_reciprocal_matrix3(self):
        c = Crystal("Natrolite(reduced cell)")
        c.lattice_angles_deg = np.fromstring("83.5  70.5  70.2", sep=' ')
        c.lattice_lengths = np.fromstring("6.601  9.739  9.893", sep=' ')
        c.calculate_reciprocal()
        isaw_rec = np.fromstring("""   0.951854   -0.000000    0.000000
  -0.342876    0.685738    0.000000
  -0.336898    0.000033    0.673719 """, sep=' ').reshape(3,3)
        #print c.reciprocal_lattice, "\n\n\n"
        assert np.allclose(isaw_rec, c.reciprocal_lattice, atol=1e-3), "Reciprocal lattice of %s matches ISAW lattice." % c.name

    def test_reciprocal_matrix4(self):
        c = Crystal("Natrolite(conventional orthorhombic cell)")
        c.lattice_angles_deg = np.fromstring("90.0  90.0  90.0", sep=' ')
        c.lattice_lengths = np.fromstring("18.328 18.585  6.597", sep=' ')
        c.calculate_reciprocal()
        isaw_rec = np.fromstring("""   0.342819    0.000000    0.000000
  -0.000000    0.338078    0.000000
  -0.000000   -0.000000    0.952431 """, sep=' ').reshape(3,3)
        #print c.reciprocal_lattice, "\n\n\n"
        assert np.allclose(isaw_rec, c.reciprocal_lattice, atol=1e-5), "Reciprocal lattice of %s matches ISAW lattice." % c.name

    #----------------------------------------------------
    #----------------------------------------------------
    def do_get_hkls_measured(self, filebase, angles):
        #@type e Experiment
        instrument.inst = instrument.Instrument("../instruments/TOPAZ_detectors_2010.csv")
        instrument.inst.goniometer = goniometer.Goniometer()
        self.exp = Experiment(instrument.inst)
        e = self.exp
        e.inst.d_min = 0.5
        e.inst.wl_min = 0.5
        e.inst.wl_max = 4.0
        e.crystal.read_ISAW_ubmatrix_file(filebase + ".mat", angles=angles)
        e.range_automatic = True
        e.range_limit_to_sphere = True
        e.initialize_reflections()
        #Position coverage object with 0 sample rotation
        poscov = instrument.PositionCoverage( angles, None, e.crystal.u_matrix)
        e.inst.positions.append(poscov)
        pos_param = ParamPositions( {id(poscov):True} )
        self.pos_param = pos_param
        e.recalculate_reflections(pos_param)
        out = []
        for ref in e.reflections:
            if ref.times_measured() > 0:
                out.append( ref.hkl )
        return out

    def test_compare_to_peaks_file(self):
        original = self.do_get_hkls_measured("data/TOPAZ_1204", [.123, np.pi/4, -0.43])
        with_rotation = self.do_get_hkls_measured("data/TOPAZ_1204", [0, 0, 0])
        assert original==with_rotation, "Same HKLs are measured with/without rotation specified."


    #----------------------------------------------------
    #----------------------------------------------------
    def dont_test_compare_ubs(self):
        """ISAWev does not take into account the goniometer angles when outputting the
        UB matrix file. ISAW does. This compares the U and B matrices obtained from both.
        """
        #0, np.pi/4, 0
        c = Crystal("oxalic acid")
        c.read_ISAW_ubmatrix_file("data/TOPAZ_1204.mat", angles=[0, np.pi/4, 0])
        c2 = Crystal("oxalic acid from EV")
        c2.read_ISAW_ubmatrix_file("data/TOPAZ_1204_ev.mat", angles=[0, 0, 0])
        assert np.allclose(c.reciprocal_lattice, c2.reciprocal_lattice, atol=0.05), "B matrices match, roughly. %s vs %s" % (c.reciprocal_lattice, c2.reciprocal_lattice)
        assert np.allclose(c.u_matrix, c2.u_matrix, atol=0.02), "U matrices match, roughly. %s vs %s" % (c.u_matrix, c2.u_matrix)
        print "\n\n\n"
        print c.u_matrix
        print c2.u_matrix
        







#==================================================================
if __name__ == "__main__":
    unittest.main()


