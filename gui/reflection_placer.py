"""GUI used to place a reflection at a particular point on a detector.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
from threading import Thread
import time
import string
import numpy as np
import sys

#--- GUI Imports ---
import gui_utils
import display_thread
import detector_plot

#--- Traits imports ---
from traits.api import HasTraits,Int,Float,Str,String,Property,Bool, List, Tuple, Array, Range, Enum
from traitsui.api import View,Item,Group,Label,Heading, Spring, Handler, TupleEditor, TabularEditor, ArrayEditor, TextEditor, CodeEditor
from traitsui.menu import OKButton, CancelButton,RevertButton
from traitsui.menu import Menu, Action, Separator

#--- Model Imports ---
import model
from model.instrument import PositionCoverage
from model.reflections import ReflectionMeasurement, Reflection


#===========================================================================
#===========================================================================
class ReflectionPlacerHandler(Handler):
    """Handler that reacts to changes in the ReflectionPlacer object."""

    #---------------------------------------------------------------------------
    def __init__(self, frame, *args, **kwargs):
        Handler.__init__(self, *args, **kwargs)
        self.add_trait('frame', frame)

    #---------------------------------------------------------------------------
    def setattr(self, info, object, name, value):
        """Called when any attribute is set."""
        #The parent class actually sets the value
        Handler.setattr(self, info, object, name, value)
        self.changed_point(object)
        #Choosing another reflection, or a different detector,
        # or switching to arbitrary mode
        if name in ["hkl", "detector", "brute_search", 
                    "arbitrary_bool", "arbitrary_xyz", "arbitrary_width"]:
            #Make sure the plot detector points to the right one
            self.frame.detectorPlot.set_detector(self.frame.placer.get_detector())
            #Make the map recalculate
            self.frame.map_thread.reset()

    #---------------------------------------------------------------------------
    def changed_point(self, object):
        #Find the orientation
        xy = object.xy.flatten()
        object.select_point(tuple (xy) )
        #Make an empty measurement point to show it
        meas = ReflectionMeasurement(None, None)
        meas.horizontal = xy[0]
        meas.vertical = xy[1]
        meas.detector_num = -1 # Mark to say NO detector
        self.frame.detectorPlot.set_measurement(meas)
        #Update other GUI
        valid_angle = not np.any(np.isnan(object.angles_deg))
        self.frame.buttonAddOrientation.Enable( valid_angle )



#===========================================================================
class DefaultEnum(Enum):
    """Traits Enum class with 'default' argument"""
    def __init__(self, *args, **kwds):
        super(DefaultEnum, self).__init__(*args, **kwds)
        if 'default' in kwds:
            self.default_value = kwds['default']




#========================================================================================================
#========================================================================================================
class PlacerMapThread(Thread):
    """Thread to calculate the allowed positions map."""

    _want_abort = False
    _do_reset = False
    
    def __init__(self, frame):
        """Constructor, also starts the thread."""
        Thread.__init__(self)
        self.frame = frame
        # This starts the thread running on creation
        self.start()

    #------------------------------------------------------------------------
    def run(self):
        """Gradually calculate the allowed map to greater resolution """
        #How many pixels to do
        (xpixels, ypixels) = [128]*2

        self._do_reset = True

        def make_image():
            """Make the image from the data last calculated."""
            #Make it all red
            buffer = np.zeros( (ypixels/step, xpixels/step, 3), dtype=np.byte)
            buffer[:,:,0]=255
            buffer[:,:,1]=200
            buffer[:,:,2]=200
            #Take only pixels every STEP pixels
            wl_okay_map = (wavelength_map[::step,::step] >= model.instrument.inst.wl_min) & (wavelength_map[::step,::step] <= model.instrument.inst.wl_max)
            #Transpose then flip to make it match the image orientation
            wl_okay_map = np.flipud(wl_okay_map.transpose())
            allowed_map = np.flipud(allowed[::step,::step].transpose())
            #Set the green pixel for anything allowed
            buffer[allowed_map, 1] = 255
            #Take out the red (making it green, yellow otherwise) if good wavelength
            buffer[wl_okay_map, 0] = 200
            #Set the not allowed areas back to pale red
            buffer[~allowed_map, 0] = 255
                        
            #Make the image
            return wx.ImageFromBuffer(xpixels/step, ypixels/step, buffer)


        while not self._want_abort:
            #Change of place?
            if self._do_reset:
                #Starting step size through the pixels
                step = 64
                #The map of allowed pixels
                allowed = np.zeros( (xpixels, ypixels), dtype=np.bool)
                calculated = np.zeros( (xpixels, ypixels), dtype=np.bool)
                block_filled = np.zeros( (xpixels, ypixels), dtype=np.bool)
                #Map of detection wavelength
                wavelength_map = np.zeros( (xpixels, ypixels), dtype=float)

                self._do_reset = False

            if step > 0:

                #The calculations are wrapped in this error handler.
                try:
                    #Calculate more!
                    t1 = time.time()
                    self.frame.placer.calculate_allowed_map(xpixels, ypixels, step,
                        allowed, wavelength_map, calculated, block_filled,
                        self.frame.placer.brute_search, callback=self.frame.calculation_callback)
                    #print "step ", step, "took ", time.time()-t1, " seconds."
                    if not self._want_abort:
                        #To avoid bug when closing
                        self.frame.detectorPlot.background_image = make_image()
                        self.frame.detectorPlot.Refresh()
                    #Reduce the step size by 2 for next time
                    step = step / 2
                    if step < 1:
                        step = 0
                        wx.CallAfter(self.frame.calculation_callback, step, 0)
                        
                except (KeyboardInterrupt, SystemExit):
                    #Allow breaking the program
                    raise
                except:
                    #Unhandled exceptions get thrown to log and message boxes.
                    (type, value, traceback) = sys.exc_info()
                    print "Exception in PlacerMapThread:\n%s\n%s\n%s" % (type, value, traceback)
                    #sys.excepthook(type, value, traceback, thread_information="reflection_placer.PlacerMapThread")
                
            else:
                #We just wait a bit
                time.sleep(0.1)
        #end of while (not aborted)

    #------------------------------------------------------------------------
    def abort(self):
        """Abort the thread. Should only be called upon exiting the reflection placer frame."""
        self.frame.placer._want_abort = True
        self._want_abort = True

    #------------------------------------------------------------------------
    def reset(self):
        """Re-calculate background."""
        #Erase the background
        self.frame.detectorPlot.background_image = None
        self.frame.detectorPlot.Refresh()
        self.frame.placer._want_abort = True
        #Make it reset
        self._do_reset = True


#===========================================================================
#===========================================================================
class ReflectionPlacer(HasTraits):
    """Class used to make GUI for the reflection placer."""
    hkl = Array( shape=(1,3), dtype=int, desc='hkl indices of the peak in question.')
    xy =  Array( shape=(1,2), dtype=float, desc='XY position on the detector face.')
    angles_deg = Array
    angles_deg_string = String
    angles_allowed = String("No")
    angles_allowed_bool = Bool(False)
    wavelength = Float(0.0)
    wavelength_can_be_measured = Bool(True)

    arbitrary_bool =  Bool(False, desc='Arbitrary direction instead of a detector?')
    arbitrary_xyz =  Array( shape=(1,3), dtype=float, desc='XYZ coordinates (in mm) to use as the direction to use. +Y = up; +Z = beam direction.')
    arbitrary_width =  Float(200, desc='When plotting the detector below, use this width and height in mm')

    starting_angles =  List( desc='Rotation matrix of the sample to be used as starting point.')
    brute_search = Bool(True)
    _want_abort = Bool(False)

    def __init__(self, refl, meas):
        """Ctor
        Parameters:
            refl: reflection object to set starting values.
            measurement_num: measurement number in refl to set starting values.
        """
        #Default array
        self.angles_deg_string = "0, 0, 0"
        #Make a list of detectors
        det_list = ["%s" % (det.name) for (i, det) in enumerate(model.instrument.inst.detectors)]
        #Create the trait that will use that list
        self.add_trait("detector", Enum( det_list ) )
        self.starting_angles = []
        if not refl is None:
            self.hkl = np.array( refl.hkl ).reshape(1,3)
            if not meas is None:
                # @type meas ReflectionMeasurement
                self.detector = det_list[meas.detector_num]
                self.xy = np.array([[ meas.horizontal, meas.vertical]])
                # @type poscov PositionCoverage
                poscov = display_thread.get_position_coverage_from_id(meas.poscov_id)
                if not poscov is None:
                    self.starting_angles = list(poscov.angles)
        self.arbitrary_xyz = np.array([500, 0.0, 0.0]).reshape(1,3)


    #------------------------------
    def calculate_allowed_map(self, xpixels, ypixels, step, allowed, wavelength_map, calculated, block_filled, brute_search, callback=None):
        """Using the current HKL, make a map of all allowed peak positios on the detector.

        Parameters:
            xpixels, ypixels: total size of the map
            step: index into the map we are calculating
            allowed, wavelength_map, calculated, block_filled: xpixels by ypixels arrays containing
                the calculation results and some markers for speeding up calcs.
            brute_search: tell the goniometer to use the brute force search; optimizer used otherwise.
            callback: GUI callback function expecting step, percent done as two inputs.
        """

        t_start = time.time()

        #Set up the parameters
        ub_matrix = model.experiment.exp.crystal.ub_matrix
        hkl = self.hkl.reshape(3,1)
        starting_angles = self.starting_angles
        if len(starting_angles)==0:
            starting_angles = None

        # @type det: FlatDetector
        det = self.get_detector()
        if det is None:
            return

        #Now we fill in the blocks that are allowed, if surrounded by allowed spots
        # t1 = time.time()
        # old_sum = np.sum(calculated)
        
        previous_step = step*2
        if previous_step <= 32:
            #Set-up block size for interpolating wavelength
            block_x = np.zeros( (previous_step, previous_step) ) + np.arange(0., previous_step)/previous_step
            block_x = block_x.transpose()
            block_y = np.zeros( (previous_step, previous_step) ) + np.arange(0., previous_step).reshape(previous_step,1)/previous_step
            block_y = block_y.transpose()
            block_xy = block_x * block_y

            for ix in xrange(1, xpixels/previous_step-2, 1):
                for iy in xrange(1, xpixels/previous_step-2, 1):
                    #Abort calculation
                    if self._want_abort:
                        self._want_abort=False
                        return

                    x = ix*previous_step
                    y = iy*previous_step
                    if not block_filled[x,y]:
                        val = allowed[x,y]
                        wl = wavelength_map[x,y]

                        should_fill_block = True
                        #This list defines an octogon of points in and around the block, from -1 to +2
                        for (dx,dy) in [(-1,0), (-1,1), (0,-1), (0,0), (0,1), (0,2), (1,-1), (1,0), (1,1), (1,2), (2,0), (2,1)]:
                            #If every "allowed" value around the block is the same, then we feel confident
                            #   that we can fill up the block.
                            should_fill_block = should_fill_block and (val == allowed[(ix+dx)*previous_step, (iy+dy)*previous_step])
                            if not should_fill_block: break

                        if should_fill_block:
                            allowed[x:x+previous_step, y:y+previous_step] = val
                            block_filled[x:x+previous_step, y:y+previous_step] = True
                            calculated[x:x+previous_step, y:y+previous_step] = True
                            #Interpolate wavelength using equation from wikipedia http://en.wikipedia.org/wiki/Bilinear_interpolation
                            #These are the 4 factors
                            b1=wl
                            b2 = wavelength_map[x+previous_step,y] - b1
                            b3 = wavelength_map[x,y+previous_step] - b1
                            b4 = b1 - wavelength_map[x+previous_step,y] - wavelength_map[x,y+previous_step] + wavelength_map[x+previous_step,y+previous_step]
                            #This applies the equation over the whole block at once. Yay Numpy!!!
                            wavelength_map[x:x+previous_step, y:y+previous_step] = \
                                b1 + b2*block_x + b3*block_y + b4*block_xy

            #print time.time()-t1, "secs to fill blocks. Went from %d to %d" % (old_sum, np.sum(calculated))

        for ix in xrange(0, xpixels, step):
            #Report progress
            if callable(callback) and (time.time()-t_start) > 0.1:
                wx.CallAfter(callback, step, ix*1.0/xpixels)
                t_start = time.time()
                
            x = (ix-xpixels/2) * det.width/(1.*xpixels)
            for iy in xrange(0, ypixels, step):
                #Don't calculate twice
                if not calculated[ix,iy]:
                    y = (iy-ypixels/2) * det.height/(1.*ypixels)
                    
                    #Abort calculation
                    if self._want_abort:
                        self._want_abort=False
                        return

                    #Pixel direction
                    beam_wanted = det.get_pixel_direction(x,y)
                    if beam_wanted is None:
                        continue

                    #The goniometer calculates the angles
                    (angles, wavelength) = \
                        model.instrument.inst.goniometer.get_sample_orientation_to_get_beam(beam_wanted, hkl, ub_matrix, starting_angles, search_method=(not brute_search))
                        
                    #Is that position allowed?
                    can_go_there = False
                    if not angles is None:
                        can_go_there = model.instrument.inst.goniometer.are_angles_allowed(angles, return_reason=False)

                    allowed[ix, iy] = can_go_there
                    calculated[ix, iy] = True
                    wavelength_map[ix, iy] = wavelength

        #To be safe
        self._want_abort = False

    #------------------------------
    def get_detector(self):
        """Return a Detector object from the selected detector."""
        if self.arbitrary_bool:
            # Build up a fake detector using the arbitrary directions
            #@type det FlatDetector
            det = model.detectors.FlatDetector("ArbitraryDetector")
            det.xpixels = 16
            det.ypixels = 16
            det.rotation = 0 #No rotation
            
            #The width and height
            det.width = self.arbitrary_width
            det.height = self.arbitrary_width
            if det.width==0.0: det.width = 1.0
            if det.height==0.0: det.height = 1.0
            
            # Get the XYZ coords of the center
            (x,y,z) = self.arbitrary_xyz[0,:]
            det.distance = np.sqrt( x*x + y*y + z*z )
            if (det.distance == 0.0):
                det.distance = 1.0
                det.azimuth_center = np.pi/2
                det.elevation_center = 0.0
            else:
                det.azimuth_center = np.arctan2(x, z)
                det.elevation_center = np.arctan(y / np.sqrt(x**2 + z**2))
                
            det.calculate_pixel_angles()
            
            # Give it back!
            return det

        else:
            # Get the name of the selected detector
            s = self.detector
            for det in model.instrument.inst.detectors:
                if s == det.name:
                    return det
        return None

    #------------------------------
    def select_point(self, det_pos):
        """Choose a point on the detector, calculate how to get there.

        Parameters:
            det_pos: position on the detector as a tuple
        """
        #Default to not found
        self.angles_deg = np.array([[np.nan, np.nan, np.nan]])
        self.angles_deg_string = "NaN"
        self.wavelength = np.nan
        #Measurement desired position
        self.xy = np.array(det_pos).reshape(1,2)
        if np.any(np.isnan(self.xy)):
            return
        # @type det: FlatDetector
        det = self.get_detector()
        if det is None:
            return
        
        #Set up the parameters
        beam_wanted = det.get_pixel_direction(det_pos[0], det_pos[1])
        if beam_wanted is None:
            return
        ub_matrix = model.experiment.exp.crystal.ub_matrix
        hkl = self.hkl.reshape(3,1)
        starting_angles = self.starting_angles
        if len(starting_angles)==0:
            starting_angles = None
        
        #The goniometer calculates the angles
        (angles, wavelength) = \
            model.instrument.inst.goniometer.get_sample_orientation_to_get_beam(beam_wanted, hkl, ub_matrix, starting_angles)

        #Save it in the object, will also show it.
        if angles is None:
            self.angles_deg = np.array( [np.nan]*3 ).reshape(1,3)
            self.angles_deg_string = "NaN"
            self.angles_allowed = "No - no angles found."
            self.angles_allowed_bool = False
            self.wavelength = np.nan
            self.wavelength_can_be_measured = False
        else:
            self.angles_deg = np.rad2deg(angles).reshape(1,len(angles))
            self.angles_deg_string =  model.instrument.inst.make_angles_string(angles)
            (allowed, reason) = model.instrument.inst.goniometer.are_angles_allowed(angles, return_reason=True)
            self.angles_allowed_bool = allowed
            if allowed:
                self.angles_allowed = "Yes!"
            else:
                self.angles_allowed = "No - " + reason
            self.wavelength = wavelength
            self.wavelength_can_be_measured = (wavelength <= model.instrument.inst.wl_max) and (wavelength >= model.instrument.inst.wl_min)

    #-------------------------------------------------------------------------------
    def add_to_list(self):
        """Adds the calculated angle set to the main list of positions."""
        #Angles in radians, as a list
        angles = np.deg2rad(self.angles_deg.flatten()).tolist()
        #Do the calculation
        poscov = model.instrument.inst.simulate_position(angles, sample_U_matrix=model.experiment.exp.crystal.get_u_matrix(), use_multiprocessing=False)
        #Make sure the position list is updated in GUI
        model.messages.send_message(model.messages.MSG_POSITION_LIST_CHANGED)
        #Add it to the list of selected items
        display_thread.select_additional_position_coverage(poscov, update_gui=True)




#===========================================================================
#===========================================================================
class FrameReflectionPlacer(wx.Frame):
    """GUI used to place a reflection at a particular point on a detector."""


    def _init_ctrls(self, prnt):
        wx.Frame.__init__(self, name=u'PanelReflectionPlacer',
              parent=prnt, pos=wx.Point(702, 235), size=wx.Size(475, 600),
              style=wx.DEFAULT_FRAME_STYLE,
              title=u'Reflection Placer')
        self.SetClientSize(wx.Size(500, 850))
        self.Bind(wx.EVT_CLOSE, self.OnFormClose)

        self.staticTextHelp = wx.StaticText( parent=self, \
            label="This window allows you to find sample orientation angles that will place the given reflection on a particular spot on a detector.")

        self.detectorPlot = detector_plot.DetectorPlot(parent=self,
              center_horizontal=True, center_vertical=True, show_coordinates=True)
        self.detectorPlot.SetBackgroundColour("white")
        self.detectorPlot.Bind(detector_plot.EVT_DETECTOR_CLICKED, self.OnDetectorClick)
#        self.detectorPlot.Bind(detector_plot.EVT_DETECTOR_CLICK_MOVED, self.OnDetectorClick)

        self.buttonAddOrientation = wx.Button(label=u'Add this orientation...', 
              parent=self, pos=wx.Point(128, 62), size=wx.Size(210, 29),
              style=0)
        self.buttonAddOrientation.Bind(wx.EVT_BUTTON, self.OnButtonAddOrientation)
        self.buttonAddOrientation.Enable(False)

        self.buttonOK = wx.Button(label=u'OK',
              parent=self, pos=wx.Point(128, 62), size=wx.Size(120, 29),
              style=0)
        self.buttonOK.Bind(wx.EVT_BUTTON, self.OnButtonOK)


        self.statusBar = wx.StatusBar(name=u'statusBar', parent=self,
              style=wx.THICK_FRAME | wx.ST_SIZEGRIP)
        self.statusBar.SetStatusText(u'Calculation status.')
        self.statusBar.SetAutoLayout(True)
        self.SetStatusBar(self.statusBar)

        #---- Sizers -----
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerAll.AddSpacer(wx.Size(8,8))
        self.boxSizerAll.AddWindow(self.staticTextHelp, 0, border=8, flag=wx.EXPAND | wx.LEFT | wx.RIGHT)
        self.boxSizerAll.AddSpacer(wx.Size(8,8))
        self.SetSizer(self.boxSizerAll)

        self.boxSizerBottom = wx.BoxSizer(orient=wx.HORIZONTAL)
        self.boxSizerBottom.AddStretchSpacer(1)
        self.boxSizerBottom.AddSpacer(wx.Size(8,8))
        self.boxSizerBottom.AddWindow(self.buttonOK, 0)



    #---------------------------------------------------------------------------
    def __init__(self, parent, refl, meas):
        self._init_ctrls(parent)
        #@type x AngleInfo
        angles_label = ", ".join([(x.name) for x in model.instrument.inst.angles])

        viewTop = View( \
            Item("hkl", label="H,K,L of the reflection:", format_str="%d"),
            Group(
                Item("arbitrary_bool", label="Use an arbitrary direction instead of a detector?")),
            Group(
                Item("arbitrary_xyz", label="Arbitrary XYZ direction:", format_str="%.2f", visible_when='arbitrary_bool'),
                Item("arbitrary_width", label="Arbitrary direction:\nWidth/height to plot:", format_str="%.2f", visible_when='arbitrary_bool')
                ),
            Item("detector", label="Detector name:", format_str="%d", visible_when='not arbitrary_bool'),
            Group(Item("xy", label="X,Y coordinates on the detector face:", format_str="%.2f")),
            Group(Label("... or use the mouse to set the position by clicking below ...")),
            )

            # Item("brute_search", label="Use brute-force search")

            
        viewBottom = View( \
            Group(Label("Measurement requires the following sample orientation:")),
            Item("angles_deg_string", label=angles_label, format_str="%s", style='readonly'),
            Item("angles_allowed", label="Sample orientation is possible?", style='readonly'),
            Item("wavelength", label="Detection wavelength in Angstroms:", format_str="%.3f", style='readonly'),
            Group(
                Label("Warning! This wavelength is outside the detector's limits!" , emphasized=True)
                , visible_when="not wavelength_can_be_measured")
            )

        self.handler = ReflectionPlacerHandler(self)
        self.placer = ReflectionPlacer(refl, meas)

        #Make sure nothing is shown there
        self.detectorPlot.background_image = None

        #Start this thread
        self.map_thread = PlacerMapThread(self)
        
        #Make it into a control
        self.controlTop = self.placer.edit_traits(parent=self, view=viewTop, kind='subpanel', handler=self.handler).control
        self.controlBottom = self.placer.edit_traits(parent=self, view=viewBottom, kind='subpanel', handler=self.handler).control

        #Put them in sizers
        self.boxSizerAll.AddWindow(self.controlTop, 0, border=4, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP)
        self.boxSizerAll.AddWindow(self.detectorPlot, 1, border=4, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP)
        self.boxSizerAll.AddWindow(self.controlBottom, 0, border=4, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP)
        self.boxSizerAll.AddSpacer(wx.Size(8,8))
        self.boxSizerAll.AddWindow(self.buttonAddOrientation, 0, border=4, flag=wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_HORIZONTAL)
        self.boxSizerAll.AddSpacer(wx.Size(8,8))
        self.boxSizerAll.AddSizer(self.boxSizerBottom, flag=wx.EXPAND)
        self.boxSizerAll.AddSpacer(wx.Size(8,8))
        self.GetSizer().Layout()

        #Make an initial update of GUIs
        self.handler.changed_point(self.placer)

    #------------------------------------------------------------
    def OnDetectorClick(self, event):
        self.placer.xy = np.array(event.data).reshape(1,2)
        self.handler.changed_point(self.placer)
        event.Skip()

    #------------------------------------------------------------
    def OnButtonAddOrientation(self, event):
        if (not self.placer.angles_allowed_bool) or (not self.placer.wavelength_can_be_measured):
            #Prompt the user to confirm, cause something is off
            s = ""
            if not self.placer.angles_allowed_bool:
                s += "\t- The goniometer cannot reach this sample orientation.\n"
            if not self.placer.wavelength_can_be_measured:
                s += "\t- The wavelength of detection, %.3f Angstroms, is outside the detector's detection limits, which are set at %.3f < wl < %.3f Angstroms.\n" % (self.placer.wavelength, model.instrument.inst.wl_min, model.instrument.inst.wl_max)
            res = wx.MessageDialog(self, "It will not be possible to measure this reflection because:\n\n" + s + "\nDo you want to add this sample orientation anyway?", "Can't Measure Reflection", wx.YES_NO | wx.YES_DEFAULT).ShowModal()
            if res == wx.ID_NO:
                event.Skip()
                return
        #User said yes or there was no problem
        self.placer.add_to_list()
        event.Skip()

    def OnButtonOK(self, event):
        self.Close()

    def OnFormClose(self, event):
        self.map_thread.abort()
        event.Skip()

    #------------------------------------------------------------
    def calculation_callback(self, step, percent):
        if step<=0:
            s = "Map calculation complete."
        else:
            s = "Calculating map, step %d, %.0f%% done."%(step, percent*100)
        self.statusBar.SetStatusText(s)
        

#===========================================================================
def show_placer_frame(parent, refl, meas):
    """Show the reflection placer frame with the given values as starting points."""
    frm = FrameReflectionPlacer(parent, refl, meas)
    frm.Show()
    return frm


#===========================================================================
if __name__ == "__main__":
    model.instrument.inst = model.instrument.Instrument(model.config.cfg.default_detector_filename)
    model.experiment.exp = model.experiment.Experiment(model.instrument.inst)
    model.experiment.exp.initialize_reflections()
    import gui_utils
    refl = Reflection( (1,1,-6), np.array([1,1,-6]) )
    (app, pnl) = gui_utils.test_my_gui(FrameReflectionPlacer, refl, None)
    app.frame.SetClientSize(wx.Size(500, 850))
    app.MainLoop()
