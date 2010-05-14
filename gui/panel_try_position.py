#Boa:FramePanel:PanelTryPosition
"""Panel to "try" to add a sample orientation to the list.
Sliders allow the user to move the sample and see the effect quickly.
When happy with the change, they can add it to the full list.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import numpy as np
import copy

#--- GUI Imports ---
import display_thread
from  value_slider import ValueSlider, EVT_VALUE_SLIDER_CHANGED, EVT_VALUE_SLIDER_CHANGING
import gui_utils

#--- Model Imports ---
import model

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
[wxID_PANELTRYPOSITION, wxID_PANELTRYPOSITIONBUTTONSAVE, 
 wxID_PANELTRYPOSITIONCHECKADD, wxID_PANELTRYPOSITIONSTATICLINETOP, 
 wxID_PANELTRYPOSITIONSTATICTEXT1, wxID_PANELTRYPOSITIONSTATICTEXTHELP, 
 wxID_PANELTRYPOSITIONSTATICTEXTWARNING, 
 wxID_PANELTRYPOSITIONSTATICTEXTWARNINGREASON, 
] = [wx.NewId() for _init_ctrls in range(8)]

#-------------------------------------------------------------------------------
class TryPositionController:
    """Controller for the PanelTryPosition class."""
    add_position = False
    angles = list([0,0,0])

    #-------------------------------------------------------------------------------
    def __init__(self, panel):
        self.panel = panel
        model.messages.subscribe( self.on_goniometer_changed, model.messages.MSG_GONIOMETER_CHANGED)


    #-------------------------------------------------------------------------------
    def set_add_position(self, value):
        """Change the setting to use the trial angles or not."""
        if value != self.add_position:
            #There was a change
            self.add_position = value
            self.redraw()

    #-------------------------------------------------------------------------------
    def on_goniometer_changed(self, *args):
        """Called when the goniometer used changes."""
        #Don't add the position now
        self.set_add_position(False)
        self.panel.checkAdd.SetValue(False)
        #Re-make all the sliders
        self.make_sliders()
        #all the angles were reset, so these need to get redone.
        self.test_angles( self._get_angles() )
        self.change_angles( self._get_angles() )

    #-------------------------------------------------------------------------------
    def make_sliders(self):
        """Generate or update all the sliders."""
        #Remove any exisiting sliders
        if hasattr(self.panel, "sliders"):
            self.panel.boxSizerSliders.Clear()
            for slid in self.panel.sliders:
                slid.Destroy()
                
        #Create all the sliders
        sliders = list()
        #Get the list of angles in this instrument.
        for angle in model.instrument.inst.angles:
            id = wx.NewId()
            # number of decimal points to keep.
            floats = 0
            if angle.friendly_units == "ang": floats=1
            slid = ValueSlider(parent=self.panel, floats=floats)
            slid.SetLabel(angle.name + " (in " + angle.friendly_units +"):")
            slid.SetMin(angle.friendly_range[0])
            slid.SetMax(angle.friendly_range[1])
            slid.SetValue(0)
            slid.Bind(EVT_VALUE_SLIDER_CHANGED, self.SliderChanged)
            slid.Bind(EVT_VALUE_SLIDER_CHANGING, self.SliderChanging)
            sliders.append(slid)
            #Add to the sizer
            self.panel.boxSizerSliders.AddWindow(slid, 0, border=0, flag=wx.EXPAND)
            self.panel.boxSizerSliders.AddSpacer(wx.Size(12, 12), border=0, flag=wx.EXPAND)
        self.panel.boxSizerSliders.Layout()
        self.panel.boxSizerAll.Layout()
        #Save to panel
        self.panel.sliders = sliders

    #-------------------------------------------------------------------------------
    def SliderChanged(self, event):
        """Called when any of the value sliders change values."""
        self.change_angles( self._get_angles() )

    def SliderChanging(self, event):
        """Called while scrolling."""
        self.test_angles( self._get_angles() )


    #-------------------------------------------------------------------------------
    def test_angles(self, angles):
        """Test the validity of the given angles."""
        #Validate the angles
        (valid, reason) = model.instrument.inst.goniometer.are_angles_allowed(angles, return_reason=True)
        if valid:
            self.panel.staticTextWarning.Hide()
            self.panel.staticTextWarningReason.Hide()
            self.panel.boxSizerAll.Layout()
        else:
            self.panel.staticTextWarning.Show()
            self.panel.staticTextWarningReason.SetLabel("Reason: %s" % reason)
            self.panel.staticTextWarningReason.Show()
            self.panel.boxSizerAll.Layout()


    #----------------------------------------------------------------------------------
    def _get_angles(self):
        """Return a list of angles in unfriendly units."""
        #Get the all the angles
        angles = list()
        i = 0
        for angle in model.instrument.inst.angles:
            slid = self.panel.sliders[i]
            #Have the angle object do the unit conversion.
            val = angle.friendly_to_internal(slid.Value)
            angles.append( val )
            i = i + 1
        return angles

    #-------------------------------------------------------------------------------
    def change_angles(self, angles):
        """Change the angles and redraw if needed."""
#        print "change_angles called with", np.rad2deg(angles)
        if self.angles != angles:
            #There was actually a change
            self.angles = angles
            if self.add_position:
                self.redraw()

    #-------------------------------------------------------------------------------
    def redraw(self):
        """Send out a command to re-do the trial addition."""
        #Create a PositionCoverage with empty qspace - means it needs to be recalculated.
        try_position = model.instrument.PositionCoverage(self.angles, None, model.experiment.exp.crystal.get_u_matrix())
        
        #Get the old try_position
        old = display_thread.get_try_position()
        if not old is None:
            if not old.try_position is None:
                #Do we have the same angles and sample orientation matrix?
                if (self.angles == old.try_position.angles) and \
                    np.allclose(model.experiment.exp.crystal.get_u_matrix(), old.try_position.sample_U_matrix):
                        #Re-use the qspace array, if it is in there.
                        try_position = model.instrument.PositionCoverage(self.angles, old.try_position.coverage, old.try_position.sample_U_matrix)
                
        #This is the try parameter
        attempt = model.experiment.ParamTryPosition(try_position, self.add_position)
        display_thread.NextParams[model.experiment.PARAM_TRY_POSITION] = attempt
        

    #-------------------------------------------------------------------------------
    def add_to_list(self):
        """Adds the last selected angle set to the main list of positions."""
        #Get the old try_position
        old = display_thread.get_try_position()
        if not old is None:
            if not old.try_position is None:
                #Do we have the same angles?
                if (self.angles == old.try_position.angles):
                    #Was it calculated?
                    if not old.try_position.coverage is None:
                        #Great! Add it to the main list.
                        #Make a copy
                        poscov = copy.copy(old.try_position)
                        model.instrument.inst.positions.append(poscov)
                        #Auto-select (check) that position we just added.
                        display_thread.select_position_coverage(poscov)
                        #Send out a message (for the GUI) saying that the list should be updated (to reflect the selection and the new item)
                        model.messages.send_message(model.messages.MSG_POSITION_LIST_CHANGED)
                        #and we're done
                        return
                    
        #If we reach here, something was missing
        wx.MessageDialog(None, "Sorry! There was a problem adding this sample orientation to the main list. Make sure it has been calculated and is shown in the coverage view before clicking this button.", 'Error', wx.OK | wx.ICON_ERROR).ShowModal()



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class PanelTryPosition(wx.Panel):
    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.staticTextHelp, 0, border=0, flag=wx.EXPAND|wx.SHRINK)
        parent.AddWindow(self.staticLineTop, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.checkAdd, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=wx.EXPAND)
        parent.AddSizer(self.boxSizerSliders, border=0, flag=wx.EXPAND)
        parent.AddWindow(self.staticTextWarning, 0, border=4,
              flag=wx.RIGHT | wx.LEFT | wx.EXPAND)
        parent.AddWindow(self.staticTextWarningReason, 0, border=16,
              flag=wx.RIGHT | wx.LEFT | wx.EXPAND)
        parent.AddSpacer(wx.Size(16, 16), border=0, flag=wx.EXPAND)
        parent.AddWindow(self.buttonSave, 0, border=0, flag=wx.EXPAND)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerSliders = wx.BoxSizer(orient=wx.VERTICAL)

        self._init_coll_boxSizerAll_Items(self.boxSizerAll)

        self.SetSizerAndFit(self.boxSizerAll)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_PANELTRYPOSITION,
              name=u'PanelTryPosition', parent=prnt, pos=wx.Point(679, 241),
              size=wx.Size(367, 523), style=wx.TAB_TRAVERSAL)
        self.SetClientSize(wx.Size(367, 523))

        self.staticTextHelp = wx.StaticText(id=wxID_PANELTRYPOSITIONSTATICTEXTHELP,
              label=u'This window allows you to try out a new sample orientation and see the change in coverage. Push the button at the bottom when you are happy with the result.',
              name=u'staticTextHelp', parent=self, pos=wx.Point(0, 0),
              style=0)
        #Mac fix for too-wide text.
        if gui_utils.is_mac():
            self.staticTextHelp.Wrap(self.GetSize()[0] - 50)

        self.staticLineTop = wx.StaticLine(id=wxID_PANELTRYPOSITIONSTATICLINETOP,
              name=u'staticLineTop', parent=self, pos=wx.Point(0, 68), style=0)

        self.checkAdd = wx.CheckBox(id=wxID_PANELTRYPOSITIONCHECKADD,
              label=u'Add this position (temporarily)', name=u'checkAdd',
              parent=self, pos=wx.Point(0, 78), style=0)
        self.checkAdd.SetValue(False)
        self.checkAdd.Bind(wx.EVT_CHECKBOX, self.OnCheckAddCheckbox,
              id=wxID_PANELTRYPOSITIONCHECKADD)

        self.buttonSave = wx.Button(id=wxID_PANELTRYPOSITIONBUTTONSAVE,
              label=u'Save this orientation in the list', name=u'buttonSave',
              parent=self, pos=wx.Point(0, 99), size=wx.Size(367, 29), style=0)
        self.buttonSave.Bind(wx.EVT_BUTTON, self.OnButtonSaveButton,
              id=wxID_PANELTRYPOSITIONBUTTONSAVE)

        self.staticTextWarning = wx.StaticText(id=wxID_PANELTRYPOSITIONSTATICTEXTWARNING,
              label=u'Warning! The angles cannot be achieved by the goniometer!',
              name=u'staticTextWarning', parent=self, pos=wx.Point(4, 100), style=0)
        self.staticTextWarning.SetForegroundColour(wx.Colour(255, 0, 0))
        self.staticTextWarning.Hide()

        self.staticTextWarningReason = wx.StaticText(id=wxID_PANELTRYPOSITIONSTATICTEXTWARNINGREASON,
              label=u'Reason: ', name=u'staticTextWarningReason', parent=self,
              pos=wx.Point(16, 134), style=0)
        self.staticTextWarningReason.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL,
              wx.BOLD, False, u'Sans'))
        self.staticTextWarningReason.SetForegroundColour(wx.Colour(255, 0, 0))
        self.staticTextWarningReason.Hide()

        self._init_sizers()

    #----------------------------------------------------------------------------------
    def __init__(self, parent):
        #wx.Panel.__init__(self, parent, id, pos, size, style, name)
        self._init_ctrls(parent)

        #Create a controller
        self.controller = TryPositionController(self)
        #Make all the slider controls using this
        self.controller.on_goniometer_changed()

    #----------------------------------------------------------------------------------

    def OnCheckAddCheckbox(self, event):
        self.controller.set_add_position(self.checkAdd.GetValue())
        event.Skip()

    def OnButtonSaveButton(self, event):
        self.controller.add_to_list()
        event.Skip()







if __name__ == '__main__':
    import gui_utils
    (app, pnl) = gui_utils.test_my_gui(PanelTryPosition)
    app.MainLoop()
