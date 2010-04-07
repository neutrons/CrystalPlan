#Boa:FramePanel:QspaceOptionsPanel
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

#--- Model Imports ---
import model


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
[wxID_QSPACEOPTIONSPANEL, wxID_QSPACEOPTIONSPANELCHECKHEMISPHERE, 
 wxID_QSPACEOPTIONSPANELCHECKINVERT, 
 wxID_QSPACEOPTIONSPANELCHECKREALTIMESLICE, 
 wxID_QSPACEOPTIONSPANELCHECKSHOWREDUNDANCY, 
 wxID_QSPACEOPTIONSPANELCHECKSHOWSLICE, 
 wxID_QSPACEOPTIONSPANELPANEL_TO_HOLD_SLICE_CONTROL, 
] = [wx.NewId() for _init_ctrls in range(7)]

class QspaceOptionsController:
    """This class is the view/controller for the QspaceOptionsPanel."""
    panel = None
    
    def __init__(self, QspaceOptionsPanel):
        """Constructor."""
        self.panel = QspaceOptionsPanel
        #Subscribe to messages
        model.messages.subscribe(self.update_data, model.messages.MSG_EXPERIMENT_QSPACE_CHANGED)

    def apply_slice(self, use_slice, slice_min, slice_max):
        """Apply a change of slicing parameters."""
        #Tell the experiment to recalculate the slice.
        display_thread.NextParams[model.experiment.PARAM_SLICE] = model.experiment.ParamSlice(use_slice, slice_min, slice_max)

    def set_invert(self, inversion):
        """Sets whether the qspace coverage should invert."""
        #Tell the experiment to invert and then recalculate the slice.
        display_thread.NextParams[model.experiment.PARAM_INVERT] = model.experiment.ParamInvert(inversion)

    def set_hemisphere(self, hemi):
        """Sets whether the coverage will use the optimal hemisphere."""
        #This will tell the display_thread what to do.
        display_thread.NextParams[model.experiment.PARAM_HEMISPHERE] = model.experiment.ParamHemisphere(hemi)

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
        


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class QspaceOptionsPanel(wx.Panel):
    """The slice panel is a custom control that allows the user to pick
    a slice through q-space to display.
    It also shows the coverage % through q-radius."""
    
    
    def _init_coll_boxSizerSliceOptions_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.checkShowSlice, 0, border=4, flag=wx.LEFT)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.checkRealtimeSlice, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=wx.LEFT)
        parent.AddWindow(self.checkInvert, 0, border=0, flag=0)

    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddSizer(self.boxSizerTop, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(3,3), border=0, flag=0)
        parent.AddSizer(self.boxSizerSliceOptions, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(4,4), border=0, flag=0)
        parent.AddWindow(self.panel_to_hold_slice_control, 1, border=4,
              flag=wx.BOTTOM | wx.RIGHT | wx.LEFT | wx.EXPAND)

    def _init_coll_boxSizerTop_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.checkHemisphere, 0, border=4,
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

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_QSPACEOPTIONSPANEL,
              name=u'QspaceOptionsPanel', parent=prnt, pos=wx.Point(676, 622),
              size=wx.Size(726, 208), style=wx.TAB_TRAVERSAL)
        self.SetClientSize(wx.Size(726, 208))
        self.SetAutoLayout(True)

        self.panel_to_hold_slice_control = wx.Panel(id=wxID_QSPACEOPTIONSPANELPANEL_TO_HOLD_SLICE_CONTROL,
              name=u'panel_to_hold_slice_control', parent=self, pos=wx.Point(4,
              53), size=wx.Size(718, 151),
              style=wx.RAISED_BORDER | wx.THICK_FRAME | wx.TAB_TRAVERSAL)
        self.panel_to_hold_slice_control.SetBackgroundColour(wx.Colour(229, 246,
              245))

        self.checkShowSlice = wx.CheckBox(id=wxID_QSPACEOPTIONSPANELCHECKSHOWSLICE,
              label=u'Show a Slice', name=u'checkShowSlice', parent=self,
              pos=wx.Point(4, 27), size=wx.Size(112, 22), style=0)
        self.checkShowSlice.SetValue(False)
        self.checkShowSlice.Bind(wx.EVT_CHECKBOX, self.OnCheckShowSliceCheckbox,
              id=wxID_QSPACEOPTIONSPANELCHECKSHOWSLICE)

        self.checkRealtimeSlice = wx.CheckBox(id=wxID_QSPACEOPTIONSPANELCHECKREALTIMESLICE,
              label=u'Real-time Update', name=u'checkRealtimeSlice',
              parent=self, pos=wx.Point(124, 27), size=wx.Size(138, 22),
              style=0)
        self.checkRealtimeSlice.SetValue(False)
        self.checkRealtimeSlice.SetMinSize(wx.Size(-1, -1))
        self.checkRealtimeSlice.Bind(wx.EVT_CHECKBOX,
              self.OnCheckRealtimeCheckbox,
              id=wxID_QSPACEOPTIONSPANELCHECKREALTIMESLICE)

        self.checkInvert = wx.CheckBox(id=wxID_QSPACEOPTIONSPANELCHECKINVERT,
              label=u'Invert (to show areas NOT covered)', name=u'checkInvert',
              parent=self, pos=wx.Point(270, 27), size=wx.Size(272, 22),
              style=0)
        self.checkInvert.SetValue(False)
        self.checkInvert.Bind(wx.EVT_CHECKBOX, self.OnCheckInvertCheckbox,
              id=wxID_QSPACEOPTIONSPANELCHECKINVERT)

        self.checkHemisphere = wx.CheckBox(id=wxID_QSPACEOPTIONSPANELCHECKHEMISPHERE,
              label=u'Find and show the optimal hemisphere',
              name=u'checkHemisphere', parent=self, pos=wx.Point(4, 0),
              size=wx.Size(296, 22), style=0)
        self.checkHemisphere.SetValue(False)
        self.checkHemisphere.Bind(wx.EVT_CHECKBOX,
              self.OnCheckHemisphereCheckbox,
              id=wxID_QSPACEOPTIONSPANELCHECKHEMISPHERE)

        self.checkShowRedundancy = wx.CheckBox(id=wxID_QSPACEOPTIONSPANELCHECKSHOWREDUNDANCY,
              label=u'Show Redundancy', name=u'checkShowRedundancy',
              parent=self, pos=wx.Point(312, 0), size=wx.Size(160, 22),
              style=0)
        self.checkShowRedundancy.SetValue(False)
        self.checkShowRedundancy.Bind(wx.EVT_CHECKBOX,
              self.OnCheckShowRedundancyCheckbox,
              id=wxID_QSPACEOPTIONSPANELCHECKSHOWREDUNDANCY)

        self._init_sizers()

    def __init__(self, parent, id, pos, size, style, name):
        self._init_ctrls(parent)
        #Create the view/controller
        self.controller = QspaceOptionsController(self)

        #Continue by adding the custom slicer control
        self.sliceControl = SliceControl(parent=self.panel_to_hold_slice_control, 
                use_slice=False, apply_slice_method=self.controller.apply_slice,
                id=wx.NewId(), name=u'panel_to_hold_slice_control', pos=wx.Point(8, 16),
                size=wx.Size(664, 100))
                        
        #Add a sizer that holds only the sliceControl, put in in the panel_to_hold_slice_control, and expand it
        sizer = wx.BoxSizer()
        sizer.Add(self.sliceControl, 1, wx.EXPAND)
        self.panel_to_hold_slice_control.SetSizer(sizer)

        #Show the initial data
        self.controller.update_data(None)
        
    def OnCheckRealtimeCheckbox(self, event):
        self.sliceControl.realtime = self.checkRealtimeSlice.GetValue()
        event.Skip()

    def OnCheckShowSliceCheckbox(self, event):
        self.sliceControl.SetUseSlice( self.checkShowSlice.GetValue() )
        event.Skip()

    def OnCheckInvertCheckbox(self, event):
        self.controller.set_invert( self.checkInvert.GetValue() )
        event.Skip()

    def OnCheckHemisphereCheckbox(self, event):
        self.controller.set_hemisphere( self.checkHemisphere.GetValue() )
        event.Skip()

    def OnCheckShowRedundancyCheckbox(self, event):
        self.controller.show_redundancy( self.checkShowRedundancy.GetValue() )
        event.Skip()



