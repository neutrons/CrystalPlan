"""Various tools and other CrystalPlan useful calculations.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id: crystal_calc.py 1325 2010-06-09 20:39:26Z 8oz $

#--- General Imports ---
from pylab import matplotlib
import numpy as np
from numpy import array, sin, cos, pi, sign
from scipy import weave
import os

#--- Model Imports ---
import experiment
import instrument
from reflections import ReflectionMeasurement
import numpy_utils
from numpy_utils import column, rotation_matrix, vector_length, normalize_vector, vector, \
                    vectors_to_matrix, az_elev_direction, within

#================================================================================
class PeakOffset:
    """Class holds info about the offset between predicted and real measurement position"""
    def __init__(self, det_num, wavelength, pred_h, pred_v, real_h, real_v):
        self.det_num = det_num
        self.predicted = np.array([pred_h, pred_v])
        self.measured = np.array([real_h, real_v])
        self.offset = self.measured-self.predicted
        self.wavelength = wavelength
        
    def __str__(self):
        return "Det %d offset of %.3f, %.3f; wl=%.3f." % (self.det_num, self.offset[0], self.offset[1], self.wavelength)

#================================================================================
def calculate_peak_offsets():
    """Calculates the offset between the predicted and the measured peak positions."""
    #@type e Experiment
    e = experiment.exp
    #@type inst Instrument
    inst = e.inst
    #@type gon Goniometer
    gon = inst.goniometer

    print len(inst.positions)
    
    offsets = []
    count = 0
    #@type ref Reflection
    for ref in e.reflections:
        ref.measurements
        #@type rrm ReflectionRealMeasurement
        for rrm in ref.real_measurements:
            for i in xrange(len(ref.measurements)):
                #@type rm ReflectionMeasurement
                rm = ReflectionMeasurement(ref, 0)
                if rm.detector_num == rrm.detector_num:
                    #rm and rrm are on the same detector.
                    #Check if the angles match
                    for poscov in inst.positions:
                        if id(poscov)==rm.poscov_id:
                            pred_angles = np.array(gon.get_phi_chi_omega(poscov.angles))
                            if np.allclose(pred_angles, np.array(rrm.angles), atol=1e-2):
                                #Great, you found a match
                                po = PeakOffset(rm.detector_num, rrm.wavelength, rm.horizontal, rm.vertical, rrm.horizontal, rrm.vertical)
                                offsets.append(po)
                                #print po

    print "%d peaks found." % (len(offsets))
    return offsets


def plot_peak_offsets(offsets):
    """Plots the results of the peak offsets calculated."""
    from pylab import *
    #@type inst Instrument
    inst = instrument.inst
    numperpage = 6
    #@type det FlatDetector
    for (det_num, det) in enumerate(inst.detectors):
        if det_num % numperpage == 0:
            figure(det_num/numperpage, figsize=[8, 10])
            clf()
            figtext(0.5, 0.95, "Offset (black line) from predicted peak positions (red dot); wavelength in angstroms", horizontalalignment='center', verticalalignment='top')
        ax = subplot(3, 2, det_num % numperpage+1)
        #Set the axes font sizes
        for xlabel_i in ax.get_xticklabels() + ax.get_yticklabels() :
            xlabel_i.set_fontsize(10)
        #@type po PeakOffset
        for po in offsets:
            if po.det_num == det_num:
                x = [po.measured[0], po.predicted[0]]
                y = [po.measured[1], po.predicted[1]]
                plot(x, y, '-k')
                plot(po.predicted[0], po.predicted[1], 'r.')
                text(po.predicted[0], po.predicted[1], ' %.1f' % po.wavelength, size=5, verticalalignment='center')
        xlim( -det.width/2, det.width/2)
        ylim( -det.height/2, det.height/2)
        #axis('equal')
        title('Detector %s' % det.name)
    #-- Save to files --
    filebase = os.path.expanduser("~") + "/peak_offsets"
    for i in xrange((len(inst.detectors) + numperpage-1) / numperpage):
        figure(i)
        savefig( filebase + "_%d.pdf" % i, papertype="letter")
    #-- combine --
    os.system("pdftk %s_*.pdf cat output %s.pdf" % (filebase, filebase))

#    show()



#================================================================================
#============================ UNIT TESTING ======================================
#================================================================================
import unittest

#==================================================================
class TestOffset(unittest.TestCase):
    """Systems test, integrating various parts of program."""
    #----------------------------------------------------
    def setUp(self):
        pass

    def test_offset(self):
        experiment.exp = experiment.load_from_file("data/TOPAZ_1241.exp")
        instrument.inst = experiment.exp.inst
        #@type e Experiment
        e = experiment.exp
        e.initialize_reflections()
        e.recalculate_reflections(e.inst.positions, None)
        self.assertEquals(len(e.reflections), 9481)
        self.assertEquals(np.sum((e.reflections_times_measured > 0)), 3427)
        offsets = calculate_peak_offsets()
        plot_peak_offsets(offsets)

#==================================================================
if __name__ == "__main__":
    unittest.main()


