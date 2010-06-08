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
class TestSystem(unittest.TestCase):
    """Systems test, integrating various parts of program."""
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

#    #----------------------------------------------------
#    #----------------------------------------------------
#    def do_get_hkls_measured(self, filebase, angles):
#        #@type e Experiment
#        instrument.inst = instrument.Instrument("../instruments/TOPAZ_detectors_2010.csv")
#        instrument.inst.goniometer = goniometer.Goniometer()
#        self.exp = Experiment(instrument.inst)
#        e = self.exp
#        e.inst.d_min = 0.5
#        e.inst.wl_min = 0.5
#        e.inst.wl_max = 4.0
#        e.crystal.read_ISAW_ubmatrix_file(filebase + ".mat", angles=angles)
#        e.range_automatic = True
#        e.range_limit_to_sphere = True
#        e.initialize_reflections()
#        #Position coverage object with 0 sample rotation
#        poscov = instrument.PositionCoverage( angles, None, e.crystal.u_matrix)
#        e.inst.positions.append(poscov)
#        pos_param = ParamPositions( {id(poscov):True} )
#        self.pos_param = pos_param
#        e.recalculate_reflections(pos_param)
#        out = []
#        for ref in e.reflections:
#            if ref.times_measured() > 0:
#                out.append( ref.hkl )
#        return out
#
#    def test_compare_to_peaks_file(self):
#        original = self.do_get_hkls_measured("data/TOPAZ_1204", [.123, np.pi/4, -0.43])
#        with_rotation = self.do_get_hkls_measured("data/TOPAZ_1204", [0, 0, 0])
#        assert original==with_rotation, "Same HKLs are measured with/without rotation specified."


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
        
    #----------------------------------------------------
    #----------------------------------------------------

    def do_test_compare_to_peaks_file(self, filebase, angles, measurement_file=None, measurement_angle=None,
                        expect_bad=0, expect_good=0):
        if measurement_angle is None:
            measurement_angle = angles
        if measurement_file is None:
            measurement_file = filebase
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
        poscov = instrument.PositionCoverage( measurement_angle, None, e.crystal.u_matrix)
        e.inst.positions.append(poscov)
        pos_param = ParamPositions( {id(poscov):True} )
        self.pos_param = pos_param
        e.recalculate_reflections(pos_param)
        #Now compare!
        (numbad, numgood, out) = e.compare_to_peaks_file(measurement_file + ".peaks")
#        print out, "\n\nsvi%d were bad; %d were good" % (numbad, numgood)
        print "%s: %d were bad; %d were good" % (filebase, numbad, numgood)
        assert numbad==expect_bad, "%s: expected %d bad peaks, but got %d bad peaks.\n%s" % (filebase, expect_bad, numbad, out)
        assert numgood==expect_good, "%s: expected %d good peaks, but got %d god peaks.\n%s" % (filebase, expect_good, numgood, out)


    def test_compare_to_peaks_file1(self):
        self.do_test_compare_to_peaks_file("data/natrolite_808_isaw", [0,0,0],
                measurement_angle=[-np.pi/6,0,0], expect_good=63, expect_bad=11)

    def test_compare_to_peaks_file2a(self):
        self.do_test_compare_to_peaks_file("data/TOPAZ_1204", [0, 0, 0],
                measurement_file="data/TOPAZ_1204", measurement_angle=[0, np.pi/4,0],
                expect_good=34, expect_bad=2)

    def test_compare_to_peaks_file2c(self):
        self.do_test_compare_to_peaks_file("data/TOPAZ_1204_ev", [0,np.pi/4,0],
                measurement_file="data/TOPAZ_1205_indexed_with_1204", measurement_angle=[np.pi/6, np.pi/4,0],
                expect_good=33, expect_bad=4)

    def test_compare_to_peaks_file2b(self):
        self.do_test_compare_to_peaks_file("data/TOPAZ_1204", [0, 0, 0],
                measurement_file="data/TOPAZ_1205_indexed_with_1204", measurement_angle=[np.pi/6, np.pi/4,0],
                expect_good=33, expect_bad=4)

    def test_compare_to_peaks_file3(self):
        self.do_test_compare_to_peaks_file("data/natrolite_807_ev", [0,0,0],
                "data/natrolite_808_indexed_with_807", [-np.pi/6,0,0],
                expect_bad=17, expect_good=57)

    def test_compare_to_peaks_file4(self):
        self.do_test_compare_to_peaks_file("data/natrolite_807_ev", [0,0,0], expect_good=39)

    def test_compare_to_peaks_file5(self):
        self.do_test_compare_to_peaks_file("data/natrolite_808_ev", [-np.pi/6,0,0], expect_good=43)

    def test_compare_to_peaks_file_omega_rotation(self):
        self.do_test_compare_to_peaks_file("data/natrolite_1223_isaw", [0,0,0],
                measurement_file="data/natrolite_1224_with_1223_mat",
                measurement_angle=np.deg2rad( [0.114, 45., 90] ), expect_good=114, expect_bad=10)


    def test_mask_error(self):
        """Error with experiment primary ref mask length not matching timesmeasured length."""
        instrument.inst = instrument.Instrument("../instruments/TOPAZ_detectors_2010.csv")
        experiment.exp = experiment.Experiment(instrument.inst)
        e = experiment.exp #@type e Experiment
        i = instrument.inst #@type i Instrument
        i.d_min = 1.0
        i.make_qspace()
        e.range_automatic = True
        e.initialize_reflections()
        e.calculate_reflections_mask()
        numref = len(e.reflections)
        param = experiment.ParamReflectionMasking(use_slice=False)
        param.primary_reflections_only = True
        e.params[experiment.PARAM_REFLECTION_MASKING] = param
        assert len(e.reflections_mask)==numref, "Correct sized mask."
        assert len(e.primary_reflections_mask)==numref, "Correct sized primary mask."
        c = e.crystal #@type c Crystal
        c.read_ISAW_ubmatrix_file("data/natrolite_1223_isaw.mat", angles=[0,0,0])
        e.initialize_reflections()
        assert len(e.reflections)>numref, "More reflections now"
        numref = len(e.reflections)
        assert len(e.primary_reflections_mask)==numref, "Correct sized primary mask."
        assert len(e.reflections_mask)==numref, "Correct sized mask."
        #Change q-space size
        i.change_qspace_size({'d_min':0.6})
        e.initialize_reflections()
        e.recalculate_reflections(None) #<--- this is necessary to make the test pass.
        #Check
        assert len(e.reflections)>numref, "Even more reflections now"
        numref = len(e.reflections)
        assert len(e.reflections_mask)==numref, "Correct sized mask."
        assert len(e.reflections_times_measured_with_equivalents)==numref, "Correct sized # of times measured."

        e.calculate_reflection_coverage_stats(True, 5.0, 5.0)
        assert e.reflection_stats_with_symmetry.total == np.sum(e.primary_reflections_mask), "Total reflection stas is the same as the # of primary reflections."
        assert e.reflection_stats_with_symmetry.measured <= e.reflection_stats_with_symmetry.total, "Coverage is less than 100%"
        assert e.reflection_stats_adjusted_with_symmetry.total == np.sum(e.primary_reflections_mask), "Adjusted: Total reflection stas is the same as the # of primary reflections."
        assert e.reflection_stats_adjusted_with_symmetry.measured <= e.reflection_stats_adjusted_with_symmetry.total, "Adjusted: Coverage is less than 100%"








#==================================================================
if __name__ == "__main__":
    unittest.main()


