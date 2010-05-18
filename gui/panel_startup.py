"""Panel to to select instrument and other options.
Most useful when starting up the application, but settings can
be changed any time.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import numpy as np

#--- GUI Imports ---
import display_thread
import gui_utils

#--- Model Imports ---
import model


# ===========================================================================================
# ===========================================================================================
# ===========================================================================================
from enthought.traits.api import HasTraits,Int,Float,Str,Property,Bool,Range
from enthought.traits.ui.api import View,Item,Label,Heading, Spring, Handler, Group
from enthought.traits.ui.menu import OKButton, CancelButton,RevertButton


# ===========================================================================================
class StartupTraitsHandler(Handler):
    """Handler that reacts to changes in the StartupParameters object."""
    frame = wx.Frame
    clicked_start = Bool(False)
    
    #----------------------------------------------------------------------
    def __init__(self, frame, *args, **kwargs):
        Handler.__init__(self, *args, **kwargs)
        self.add_trait('frame', frame)
        self.add_trait('instrument', "")
        #Make it change the linked object to reflect the current instrument state.
        self.revert()
        self.check_validity()

    #----------------------------------------------------------------------
    def setattr(self, info, object, name, value):
        """Called when any attribute is set."""
        Handler.setattr(self, info, object, name, value)
        self.check_validity()

    #----------------------------------------------------------------------
    def check_validity(self):
        par = self.frame.params #@type par StartupParameters
        
        #Try to adjust the points?
        if par.keep_points_same:
            qr = (par.q_lim*2) /(par.points_goal**(1./3))
            par.q_resolution = qr

        #Show a warning
        if par.points > 3e6:
            wx.CallAfter(self.frame.staticTextSpaceWarning.Show)
        else:
            wx.CallAfter(self.frame.staticTextSpaceWarning.Hide)
        wx.CallAfter(self.frame.GetSizer().Layout)

        #Disable the start button if too few points come up
        wx.CallAfter(self.frame.buttonApply.Enable, self.frame.params.points > 100 )

    #----------------------------------------------------------------------
    def apply(self):
        """Apply changes now."""
        #This sets up the new size q-space
        model.instrument.inst.change_qspace_size(self.get_params_dictionary())

        #Fix the hkl range if needed
        if  model.experiment.exp.range_automatic:
            model.experiment.exp.initialize_reflections()
        
        #This recalcs the volumes
        gui_utils.do_recalculation_with_progress_bar(new_sample_U_matrix=None)

        #Ensure that the q-space viewer refreshes properly.
        display_thread.handle_change_of_qspace()

        #Save the points goal
        par = self.frame.params #@type par StartupParameters
        par.points_goal = par.points


    #----------------------------------------------------------------------
    def revert(self):
        """Revert to the original values."""
        #Populate with the current values in the global instrument
        self.frame.params.set_params(model.instrument.inst)
        #Hide/show warning
        self.check_validity()

    #----------------------------------------------------------------------
    def get_params_dictionary(self):
        """Returns a dictionary of the useful parameters saved."""
        params = {'instrument':self.instrument}
        #Merge with the ones from the StartupParameters object
        params.update(self.frame.params.get_params())
        return params




# ===========================================================================================
class StartupParameters(HasTraits):
    """This traits object holds parameters to start-up the program,
    or when re-calculating the q-space."""
    d_min = Float(1.0)
    q_resolution = Float(0.2)
    d_max = Str(" +infinity ")
    #Detector wavelength limits in A
    wl_min = Range(1e-3, +np.inf, 0.5, exclude_low=False)
    wl_max = Range(1e-3, +np.inf, 4.0, exclude_low=False)

    keep_points_same = Bool(True, label='Keep # of points ~constant?', desc="to adjust the resolution so as to keep the # of points in 3D approximately constant.")
    points_goal = Float(250000, label='Target # of points:', desc="the desired number of points (3D voxels). The resolution will be adjusted to match." )

    #q-space limit property
    q_lim = Property(Float,depends_on=["d_min"])
    def _get_q_lim(self):
        if  self.d_min < 1e-3:
            return np.nan
        else:
            return 2*np.pi / self.d_min

    #number of points
    points = Property(Int,depends_on=["d_min", "q_resolution"])
    def _get_points(self):
        if  self.d_min < 1e-3 or self.q_resolution < 1e-3:
            return 0
        else:
            return round(2*2*np.pi / self.d_min/ self.q_resolution)**3


    #Create the view
    header_text = Group(Spring(label="Enter the parameters for Q-space simulation:", emphasized=True, show_label=True))
    second_label = Group(Spring(label="Enter the detectors' wavelength limits:", emphasized=True, show_label=True))
    third_label = Group(Spring(label="Note: wavelength and bandwidth can also be set, on option, as a goniometer motor setting. See the goniometer tab.", emphasized=False, show_label=True))

    view = View( header_text,
                 Item("d_min", label="d_min (angstroms)", format_str="%.3f", tooltip="Minimum d spacing to simulate."),
                 Item("d_max", label="d_max (angstroms)", style='readonly', tooltip="Maximum d spacing to simulate."),
                 Item("q_lim",  label="Resulting q-space range is:", style='readonly', format_str="+-%.3f angstroms^-1", enabled_when="True"),
                 Item("q_resolution",  label="Resolution in q-space (angstroms^-1)", format_str="%.3f", enabled_when="not keep_points_same"),
                 Item("keep_points_same"),
                 Item("points_goal", format_str="%.0f", enabled_when="keep_points_same"),
                 Item("points",  label="Number of points in space:", style='readonly', format_func=gui_utils.print_large_number),
                 second_label,
                 third_label,
                 Item("wl_min", label="Min. wavelength (angstroms)", format_str="%.3f", tooltip="Minimum wavelength that the detectors can measure."),
                 Item("wl_max", label="Max. wavelength (angstroms)", format_str="%.3f", tooltip="Maximum wavelength that the detectors can measure."),
                 kind='panel',
            )
            

    useful_list = ['d_min', 'q_resolution', 'wl_min', 'wl_max']
    
    def get_params(self):
        """Returns a dictionary of the useful parameters saved."""
        params = {}
        for name in self.useful_list:
            if hasattr(self, name):
                params[name] = getattr(self, name)
        return params


    def set_params(self, other):
        """Sets the attributes of this object to match those of the passed in object."""
        for name in self.useful_list:
            if hasattr(other, name):
                setattr(self, name,  getattr(other, name) )
        


# ===========================================================================================
# ===========================================================================================
# ===========================================================================================

[wxID_DIALOGSTARTUP, wxID_DIALOGSTARTUPBUTTONQUIT, 
 wxID_DIALOGSTARTUPbuttonApply, wxID_DIALOGSTARTUPGAUGERECALC,
 wxID_DIALOGSTARTUPLISTINSTRUMENTS, wxID_DIALOGSTARTUPSTATICLINE1, 
 wxID_DIALOGSTARTUPSTATICLINE2, wxID_DIALOGSTARTUPSTATICTEXTHELP, 
 wxID_DIALOGSTARTUPSTATICTEXTRECALCULATIONPROGRESS, 
 wxID_DIALOGSTARTUPSTATICTEXTRECALCULATIONSTATUS, 
 wxID_DIALOGSTARTUPSTATICTEXTSELECT, wxID_DIALOGSTARTUPSTATICTEXTSPACER1, 
 wxID_DIALOGSTARTUPSTATICTEXTSPACEWARNING, wxID_DIALOGSTARTUPSTATICTEXTTITLE, 
] = [wx.NewId() for _init_ctrls in range(14)]

class PanelStartup(wx.Panel):
    def _init_coll_boxSizerList_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)
        parent.AddWindow(self.listInstruments, 2, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)

    def _init_coll_boxSizerButtons_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)
        parent.AddWindow(self.buttonApply, 0, border=0, flag=0)
        parent.AddWindow(self.staticTextSpacer1, 1, border=0, flag=0)
        parent.AddWindow(self.buttonQuit, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)

    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextHelp, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextSelect, 0, border=0, flag=wx.EXPAND)
        parent.AddSizer(self.boxSizerList, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(24, 8), border=0, flag=0)
        parent.AddWindow(self.staticLine1, 0, border=0, flag=wx.EXPAND)
        parent.AddSizer(self.boxSizerParams, 1, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextSpaceWarning, 0, border=0,
              flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticLine2, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddSizer(self.boxSizerButtons, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)

        self.boxSizerList = wx.BoxSizer(orient=wx.HORIZONTAL)

        self.boxSizerParams = wx.BoxSizer(orient=wx.VERTICAL)

        self.boxSizerButtons = wx.BoxSizer(orient=wx.HORIZONTAL)

        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerList_Items(self.boxSizerList)
        self._init_coll_boxSizerButtons_Items(self.boxSizerButtons)

        self.SetSizer(self.boxSizerAll)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_DIALOGSTARTUP, name=u'PanelStartup',
              parent=prnt, pos=wx.Point(702, 235), size=wx.Size(600, 600))
        self.SetClientSize(wx.Size(600, 600))

        self.staticTextHelp = wx.StaticText(id=wxID_DIALOGSTARTUPSTATICTEXTHELP,
              label=u"The CrystalPlan application is used to simulate the coverage of reciprocal space of an instrument's detectors, given a list of sample orientations.",
              name=u'staticTextHelp', parent=self, pos=wx.Point(0, 40),
              style=0)
        if gui_utils.is_mac():
            self.staticTextHelp.Wrap(self.GetSize()[0]-50)

        self.staticTextSelect = wx.StaticText(id=wxID_DIALOGSTARTUPSTATICTEXTSELECT,
              label=u'Please select the instrument you will be simulating:',
              name=u'staticTextSelect', parent=self, pos=wx.Point(0, 99),
              size=wx.Size(475, 17), style=0)

        self.listInstruments = wx.ListBox(choices=['TOPAZ', 'Other Instruments...'],
              id=wxID_DIALOGSTARTUPLISTINSTRUMENTS, name=u'listInstruments',
              parent=self, pos=wx.Point(16, 116), size=wx.Size(443, 149),
              style=0)

        self.staticLine1 = wx.StaticLine(id=wxID_DIALOGSTARTUPSTATICLINE1,
              name='staticLine1', parent=self, pos=wx.Point(0, 273),
              size=wx.Size(475, 2), style=0)

        self.buttonApply = wx.Button(id=wxID_DIALOGSTARTUPbuttonApply,
              label=u'  Apply Changes  ', name=u'buttonApply', parent=self,
              pos=wx.Point(16, 563), style=0)
        self.buttonApply.Bind(wx.EVT_BUTTON, self.OnbuttonApplyButton,
              id=wxID_DIALOGSTARTUPbuttonApply)

        self.buttonQuit = wx.Button(id=wxID_DIALOGSTARTUPBUTTONQUIT,
              label=u'  Revert  ', name=u'buttonQuit', parent=self, pos=wx.Point(309,
              563), style=0)
        self.buttonQuit.Bind(wx.EVT_BUTTON, self.OnButtonQuitButton,
              id=wxID_DIALOGSTARTUPBUTTONQUIT)

        self.staticTextSpacer1 = wx.StaticText(id=wxID_DIALOGSTARTUPSTATICTEXTSPACER1,
              label=u' ', name=u'staticTextSpacer1', parent=self,
              pos=wx.Point(166, 563), size=wx.Size(320, 17), style=0)

        self.staticLine2 = wx.StaticLine(id=wxID_DIALOGSTARTUPSTATICLINE2,
              name='staticLine2', parent=self, pos=wx.Point(0, 491),
              size=wx.Size(478, 2), style=0)

        self.staticTextSpaceWarning = wx.StaticText(id=wxID_DIALOGSTARTUPSTATICTEXTSPACEWARNING,
              label=u'Warning: This many points may make calculations slow. Increase the q-space resolution, or reduce the q-space range by increasing d_min.',
              name=u'staticTextSpaceWarning', parent=self, pos=wx.Point(0, 432),
              size=wx.Size(475, 51), style=0)
        self.staticTextSpaceWarning.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL,
              wx.NORMAL, False, u'Sans'))
        self.staticTextSpaceWarning.SetForegroundColour(wx.Colour(255, 0, 0))


        self._init_sizers()

    def __init__(self, parent):
        #recalc_mode is True when the window is to be used to recalculate q-space, rather than start the program.
        self._init_ctrls(parent)

        #Hide the warning initially
        self.staticTextSpaceWarning.Hide()

        #Hide the instrument stuff, since that does nothing now
        self.staticTextSelect.Hide()
        self.listInstruments.Hide()

        #Setup the parameter editor traits ui panel
        self.params = StartupParameters()
        if not model.instrument.inst is None:
            if not model.instrument.inst.qspace_radius is None:
                self.params.points_goal = model.instrument.inst.qspace_radius.size
                
        self.handler = StartupTraitsHandler(self)
        self.control = self.params.edit_traits(parent=self, kind='subpanel', handler=self.handler).control
        self.boxSizerParams.AddWindow(self.control, 3, border=1, flag=wx.EXPAND)
        self.GetSizer().Layout()


    def OnbuttonApplyButton(self, event):
        self.handler.apply()
        event.Skip()

    def OnButtonQuitButton(self, event):
        self.handler.revert()
        event.Skip()



# ===========================================================================================
# ===========================================================================================
# ===========================================================================================

if __name__ == '__main__':
    import gui_utils
    (app, pnl) = gui_utils.test_my_gui(PanelStartup)
    app.frame.SetClientSize(wx.Size(700,500))
    app.MainLoop()

    
