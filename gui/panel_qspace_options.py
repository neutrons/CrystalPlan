#Boa:FramePanel:PanelQspaceOptions
"""Panel showing options on how to display reciprocal-space
volume coverage."""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import numpy as np

#--- GUI Imports ---
from slice_control import SliceControl
import display_thread
import config_gui
import gui_utils

#--- Model Imports ---
import model


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
[wxID_PanelQspaceOptions, wxID_PanelQspaceOptionsCHECKSYMMETRY,
 wxID_PanelQspaceOptionsCHECKINVERT,
 wxID_PanelQspaceOptionsCHECKREALTIMESLICE,
 wxID_PanelQspaceOptionsCHECKSHOWREDUNDANCY,
 wxID_PanelQspaceOptionsCHECKSHOWSLICE,
 wxID_PanelQspaceOptionsPANEL_TO_HOLD_SLICE_CONTROL,
] = [wx.NewId() for _init_ctrls in range(7)]

class QspaceOptionsController:
    """This class is the view/controller for the PanelQspaceOptions."""
    panel = None
    
    def __init__(self, PanelQspaceOptions):
        """Constructor."""
        self.panel = PanelQspaceOptions
        #Subscribe to messages
        model.messages.subscribe(self.update_data, model.messages.MSG_EXPERIMENT_QSPACE_CHANGED)

    def apply_slice(self, use_slice, slice_min, slice_max):
        """Apply a change of slicing parameters."""
        #Tell the experiment to recalculate the slice.
        display_thread.NextParams[model.experiment.PARAM_SLICE] = model.experiment.ParamSlice(use_slice, slice_min, slice_max)

    def apply_energy_slice(self, use_slice, slice_min, slice_max):
        """Apply a change of energy slicing parameters."""
        #Tell the experiment to recalculate the ENERGY slice.
        display_thread.NextParams[model.experiment.PARAM_ENERGY_SLICE] = model.experiment.ParamSlice(use_slice, slice_min, slice_max)

    def set_invert(self, inversion):
        """Sets whether the qspace coverage should invert."""
        #Tell the experiment to invert and then recalculate the slice.
        display_thread.NextParams[model.experiment.PARAM_INVERT] = model.experiment.ParamInvert(inversion)

    def set_symmetry(self, value):
        """Sets whether the coverage will use the crystal symmetry."""
        #This will tell the display_thread what to do.
        display_thread.NextParams[model.experiment.PARAM_SYMMETRY] = model.experiment.ParamSymmetry(value)

    def show_redundancy(self, value):
        """Sets whether the redundancy is displayed graphically (using transparent isosurfaces)."""
        display_thread.show_redundancy(value)
    
    def __del__(self):
        self.cleanup()
        
    def cleanup(self):
        """Clean-up routine for closing the view."""
        model.messages.unsubscribe(self.update_data)
        
    def update_data(self, argument):
        """Called when a message is received saying that the q-space calculation has changed. 
        Will update the graphical display.
            argument: ignored; was necessary for the pubsub message passing system."""
        #print "SliceController.update_data"
        
        #Also do the coverage plot
        (data_x, data_y) = model.experiment.exp.get_coverage_stats_data()
        self.panel.sliceControl.SetData(data_x, data_y)
        #And redraw it
        self.panel.sliceControl.Refresh()

        if gui_utils.inelastic_mode():
            #Now do the energy slice data
            #TODO: Un-fake data
            data_x = np.arange(-50, 20, 1)
            #data_y = [(1.0, 0., 0., 0., 0.) for x in data_x]
            data_y = []
            for i in xrange(4):
                data_y.append( data_x*0 )
            self.panel.sliceEnergy.SetData(data_x, data_y)
            self.panel.sliceEnergy.Refresh()



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class PanelQspaceOptions(wx.Panel):
    """The slice panel is a custom control that allows the user to pick
    a slice through q-space to display.
    It also shows the coverage % through q-radius."""
    
    
    def _init_coll_boxSizerSliceOptions_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.checkShowSlice, 0, border=4, flag=wx.LEFT)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.checkInvert, 0, border=0, flag=0)
        parent.AddStretchSpacer(1)
        parent.AddWindow(self.checkRealtimeSlice, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)

    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddSizer(self.boxSizerTop, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(3,3), border=0, flag=0)
        parent.AddSizer(self.boxSizerSliceOptions, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(4,4), border=0, flag=0)
        parent.AddWindow(self.panel_to_hold_slice_control, 1, border=4,
              flag=wx.BOTTOM | wx.RIGHT | wx.LEFT | wx.EXPAND)

    def _init_coll_boxSizerTop_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.checkSymmetry, 0, border=4,
              flag=wx.LEFT | wx.RIGHT)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.checkShowRedundancy, 0, border=0, flag=0)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)

        self.boxSizerTop = wx.BoxSizer(orient=wx.HORIZONTAL)

        self.boxSizerSliceOptions = wx.BoxSizer(orient=wx.HORIZONTAL)

        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerTop_Items(self.boxSizerTop)
        self._init_coll_boxSizerSliceOptions_Items(self.boxSizerSliceOptions)

        self.SetSizer(self.boxSizerAll)

    #----------------------------------------------------------------------------
    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_PanelQspaceOptions,
              name=u'PanelQspaceOptions', parent=prnt, pos=wx.Point(676, 622),
              size=wx.Size(726, 208), style=wx.TAB_TRAVERSAL)
        self.SetClientSize(wx.Size(726, 208))
        self.SetAutoLayout(True)

        self.panel_to_hold_slice_control = wx.Panel(id=wxID_PanelQspaceOptionsPANEL_TO_HOLD_SLICE_CONTROL,
              name=u'panel_to_hold_slice_control', parent=self, pos=wx.Point(4,
              53), size=wx.Size(718, 151),
              style=wx.RAISED_BORDER | wx.THICK_FRAME | wx.TAB_TRAVERSAL)

        self.checkShowSlice = wx.CheckBox(id=wxID_PanelQspaceOptionsCHECKSHOWSLICE,
              label=u'Show a slice in q', name=u'checkShowSlice', parent=self,
              pos=wx.Point(4, 27),  style=0)
        self.checkShowSlice.SetValue(False)
        self.checkShowSlice.Bind(wx.EVT_CHECKBOX, self.OnCheckShowSliceCheckbox,
              id=wxID_PanelQspaceOptionsCHECKSHOWSLICE)

        self.checkRealtimeSlice = wx.CheckBox(id=wxID_PanelQspaceOptionsCHECKREALTIMESLICE,
              label=u'Real-time slice', name=u'checkRealtimeSlice',
              parent=self, pos=wx.Point(124, 27), style=0)
        self.checkRealtimeSlice.SetValue(False)
        self.checkRealtimeSlice.SetMinSize(wx.Size(-1, -1))
        self.checkRealtimeSlice.Bind(wx.EVT_CHECKBOX,
              self.OnCheckRealtimeCheckbox,
              id=wxID_PanelQspaceOptionsCHECKREALTIMESLICE)

        self.checkInvert = wx.CheckBox(id=wxID_PanelQspaceOptionsCHECKINVERT,
              label=u'Invert (to show areas NOT covered)', name=u'checkInvert',
              parent=self, pos=wx.Point(270, 27), size=wx.Size(272, 22),
              style=0)
        self.checkInvert.SetValue(False)
        self.checkInvert.Bind(wx.EVT_CHECKBOX, self.OnCheckInvertCheckbox,
              id=wxID_PanelQspaceOptionsCHECKINVERT)

        self.checkSymmetry = wx.CheckBox(id=wxID_PanelQspaceOptionsCHECKSYMMETRY,
              label=u'Use crystal symmetry?   ',
              name=u'checkSymmetry', parent=self, pos=wx.Point(4, 0), style=0)
        self.checkSymmetry.SetValue(False)
        self.checkSymmetry.Bind(wx.EVT_CHECKBOX,
              self.OnCheckSymmetryCheckbox,
              id=wxID_PanelQspaceOptionsCHECKSYMMETRY)

        self.checkShowRedundancy = wx.CheckBox(id=wxID_PanelQspaceOptionsCHECKSHOWREDUNDANCY,
              label=u'Show Redundancy', name=u'checkShowRedundancy',
              parent=self, pos=wx.Point(312, 0), size=wx.Size(160, 22),
              style=0)
        self.checkShowRedundancy.SetValue(False)
        self.checkShowRedundancy.Bind(wx.EVT_CHECKBOX,
              self.OnCheckShowRedundancyCheckbox,
              id=wxID_PanelQspaceOptionsCHECKSHOWREDUNDANCY)

        self._init_sizers()

    #----------------------------------------------------------------------------
    def __init__(self, parent):
        self._init_ctrls(parent)
        #Create the view/controller
        self.controller = QspaceOptionsController(self)

        #Continue by adding the custom slicer control
        self.sliceControl = SliceControl(parent=self.panel_to_hold_slice_control,
                use_slice=False, apply_slice_method=self.controller.apply_slice,
                id=wx.NewId(), name=u'panel_to_hold_slice_control', pos=wx.Point(8, 16),
                size=wx.Size(664, 100))

        self.sizerSlices = wx.BoxSizer(orient=wx.VERTICAL)
        self.sizerSlices.Add(self.sliceControl, 1, wx.EXPAND)
        self.panel_to_hold_slice_control.SetSizer(self.sizerSlices)

        #Finish up
        if gui_utils.inelastic_mode():
            self._init_inelastic()

        #Show the initial data
        self.controller.update_data(None)


    #----------------------------------------------------------------------------
    def _init_inelastic(self):
        """Initialization of controls for inelastic mode."""

        #Continue by adding the custom slicer control
        self.sliceEnergy = SliceControl(parent=self.panel_to_hold_slice_control,
                use_slice=False, apply_slice_method=self.controller.apply_energy_slice,
                id=wx.NewId(), pos=wx.Point(8, 16),
                size=wx.Size(664, 100))
        self.sliceEnergy.use_slice = True
        self.sliceEnergy.energy_mode = True

        self.staticTextEnergySlice = wx.StaticText(id=wx.NewId(),
              label="Select slice of neutron energy change (in meV):", parent=self.panel_to_hold_slice_control,
              style=0)

        #Add to sizer
        self.sizerSlices.Add(self.staticTextEnergySlice, 0, border=4, flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.BOTTOM)
        self.sizerSlices.Add(self.sliceEnergy, 1, wx.EXPAND)


    #----------------------------------------------------------------------------
    def OnCheckRealtimeCheckbox(self, event):
        self.sliceControl.realtime = self.checkRealtimeSlice.GetValue()
        event.Skip()

    def OnCheckShowSliceCheckbox(self, event):
        self.sliceControl.SetUseSlice( self.checkShowSlice.GetValue() )
        event.Skip()

    def OnCheckInvertCheckbox(self, event):
        self.controller.set_invert( self.checkInvert.GetValue() )
        event.Skip()

    def OnCheckSymmetryCheckbox(self, event):
        self.controller.set_symmetry( self.checkSymmetry.GetValue() )
        event.Skip()

    def OnCheckShowRedundancyCheckbox(self, event):
        self.controller.show_redundancy( self.checkShowRedundancy.GetValue() )
        event.Skip()





# ===========================================================================================
# ===========================================================================================
# ===========================================================================================

if __name__ == '__main__':
    #Ok, create the instrument
    model.instrument.inst = model.instrument.InstrumentInelastic("../instruments/TOPAZ_detectors_2010.csv")
    model.instrument.inst.make_qspace()
    #Initialize the instrument and experiment
    model.experiment.exp = model.experiment.Experiment(model.instrument.inst)
    import gui_utils
    (app, pnl) = gui_utils.test_my_gui(PanelQspaceOptions)
    app.frame.SetClientSize(wx.Size(700,500))
    app.MainLoop()


