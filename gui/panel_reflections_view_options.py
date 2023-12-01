#Boa:FramePanel:PanelReflectionsViewOptions
"""Panel to control options on how to view single reflections
in the 3D reciprocal space viewer.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx

#--- GUI Imports ---
import display_thread
from slice_control import SliceControl

#--- Model Imports ---
import model


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

[wxID_PANELREFLECTIONSVIEWOPTIONS, 
 wxID_PANELREFLECTIONSVIEWOPTIONSCHECKAUTOSIZE, 
 wxID_PANELREFLECTIONSVIEWOPTIONSCHECKHIGHLIGHTIMPORTANT, 
 wxID_PANELREFLECTIONSVIEWOPTIONSCHECKREALTIMESLICE, 
 wxID_PANELREFLECTIONSVIEWOPTIONSCHECKSHOWSLICE, 
 wxID_PANELREFLECTIONSVIEWOPTIONSCHOICEVIEW, 
 wxID_PANELREFLECTIONSVIEWOPTIONSPANEL_TO_HOLD_SLICE_CONTROL, 
 wxID_PANELREFLECTIONSVIEWOPTIONSRADIOPIXELS, 
 wxID_PANELREFLECTIONSVIEWOPTIONSRADIOSPHERES, 
 wxID_PANELREFLECTIONSVIEWOPTIONSSLIDERSIZE, 
 wxID_PANELREFLECTIONSVIEWOPTIONSSTATICTEXTDISPLAYAS, 
 wxID_PANELREFLECTIONSVIEWOPTIONSSTATICTEXTSIZE, 
 wxID_PANELREFLECTIONSVIEWOPTIONSSTATICTEXTVIEWOPTION, 
] = [wx.NewId() for _init_ctrls in range(13)]


#The options shown in the drop-down box.
ReflectionsViewChoices = ['All reflections',
    'Predicted reflections',
    'NON-predicted reflections',
    'Measured reflections',
    'NON-measured reflections',
    'Predicted but not measured',
    'Measured but not predicted']

#-------------------------------------------------------------------------------
class ReflectionsViewOptionsController:
    """This class is the view/controller for the PanelReflectionsViewOptions."""
    # @type panel PanelReflectionsViewOptions
    panel = None

    _inside_show_settings = False

    #-------------------------------------------------------------------------------
    def __init__(self, panel):
        """Constructor."""
        self.panel = panel
        #Subscribe to messages
        model.messages.subscribe(self.update_data, model.messages.MSG_EXPERIMENT_QSPACE_CHANGED)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """Clean-up routine for closing the view."""
        model.messages.unsubscribe(self.update_data, model.messages.MSG_EXPERIMENT_QSPACE_CHANGED)


    #-------------------------------------------------------------------------------
    def show_settings(self):
        """Change the UI elements to match the current display settings and other options.
        Should be called when creating the form.
        """
        self._inside_show_settings = True
        #- Display settings
        # @type param ParamReflectionDisplay
        param = display_thread.get_reflection_display_params()
        self.panel.radioPixels.SetValue(param.display_as == param.DISPLAY_AS_PIXELS)
        self.panel.radioSpheres.SetValue(param.display_as == param.DISPLAY_AS_SPHERES)
        self.panel.radioMeasured.SetValue(param.color_map == param.COLOR_BY_MEASURED)
        self.panel.radioPredicted.SetValue(param.color_map == param.COLOR_BY_PREDICTED)
        self.panel.sliderSize.SetValue(10. * param.size)
        self.panel.sliderSize.Enable(not param.automatic_size)
        self.panel.checkAutoSize.SetValue(param.automatic_size)
        
        #- Masking settings
        # @type param ParamReflectionMasking
        param = display_thread.get_reflection_masking_params()
        self.panel.choiceView.Select(param.masking_type)
        self.panel.textThreshold.SetValue( "%.2f" % param.threshold)
        #Slice settings
        self.panel.checkShowSlice.SetValue(param.use_slice)
        self.panel.sliceControl.use_slice = param.use_slice
        self.panel.sliceControl.slice_min = param.slice_min
        self.panel.sliceControl.slice_max = param.slice_max
        self._inside_show_settings = False

    #-------------------------------------------------------------------------------
    def change_displayed_size(self, newsize):
        """Change the pixel size shown in the UI. DO NOT apply the change."""
        self._inside_show_settings = True
        self.panel.sliderSize.SetValue(newsize * 10.)
        self._inside_show_settings = False
        
    #-------------------------------------------------------------------------------
    def change_masking_settings(self):
        """Call whenever a checbkox changes the masking settings."""
        if self._inside_show_settings: return
        # @type ctl SliceControl
        ctl = self.panel.sliceControl
        mask = model.experiment.ParamReflectionMasking(self.panel.checkShowSlice.GetValue()>0, ctl.slice_min, ctl.slice_max)
        #Continue setting
        self._set_masking_settings(mask)

    #-------------------------------------------------------------------------------
    def change_display_settings(self):
        """Call when any of the display options (pixels/spheres) change."""
        if self._inside_show_settings: return
        # @type param ParamReflectionDisplay
        param = model.experiment.ParamReflectionDisplay()
        #Set the values
        if self.panel.radioPixels.GetValue() > 0:
            param.display_as = param.DISPLAY_AS_PIXELS
        else:
            param.display_as = param.DISPLAY_AS_SPHERES
            
        if self.panel.radioMeasured.GetValue() > 0:
            param.color_map = param.COLOR_BY_MEASURED
        else:
            param.color_map = param.COLOR_BY_PREDICTED
            
        param.size = self.panel.sliderSize.GetValue() / 10.
        param.automatic_size = self.panel.checkAutoSize.GetValue()
        #Make sure the slider is enabled/disabled as needed
        self.panel.sliderSize.Enable(not param.automatic_size)

        #Trigger the update
        display_thread.NextParams[model.experiment.PARAM_REFLECTION_DISPLAY] = param

    #-------------------------------------------------------------------------------
    def _set_masking_settings(self, mask):
        """Set the masking settings using the checkbox etc. choices.

        Parameters:
            mask: a ParamReflectionMasking object."""
        n = self.panel.choiceView.GetSelection()
        #@type mask ParamReflectionMasking
        mask.masking_type = n
        mask.primary_reflections_only = self.panel.checkUseSymmetry.GetValue()
        mask.show_equivalent_reflections = self.panel.checkUseSymmetry.GetValue()
        try:
            mask.threshold = float(self.panel.textThreshold.GetValue())
        except:
            mask.threshold = -1.0
        #Apply it.
        display_thread.NextParams[model.experiment.PARAM_REFLECTION_MASKING] = mask

    #-------------------------------------------------------------------------------
    def apply_slice(self, use_slice, slice_min, slice_max):
        """Apply a change of slicing parameters."""
        #Tell the experiment to recalculate the slice.
        mask = model.experiment.ParamReflectionMasking(use_slice, slice_min, slice_max)
        self._set_masking_settings(mask)

    #-------------------------------------------------------------------------------
    def update_data(self, argument):
        """Called when a message is received saying that the q-space calculation has changed.
        Will update the graphical display.
            argument: ignored; was necessary for the pubsub message passing system."""
        #Do the coverage plot
        (data_x, data_y) = model.experiment.exp.get_coverage_stats_data()
        self.panel.sliceControl.SetData(data_x, data_y)
        #And redraw it
        self.panel.sliceControl.Refresh()




#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class PanelReflectionsViewOptions(wx.Panel):
    def _init_coll_boxSizerSliceOptions_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.checkShowSlice, 0, border=4,
              flag=wx.LEFT | wx.SHRINK)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.checkRealtimeSlice, 0, border=0, flag=0)

    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(4,4), border=0, flag=0)
        parent.AddSizer(self.boxSizerTop, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(4, 4), border=0, flag=0)
        parent.AddSizer(self.boxSizerColor, 0, border=0, flag=wx.EXPAND)
        parent.AddSizer(self.boxSizerDisplay, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(4, 4), border=0, flag=0)
        parent.AddSizer(self.boxSizerSliceOptions, 0, border=0, flag=0)
        self.boxSizerAll.AddWindow(self.panel_to_hold_slice_control, 1,
              border=4, flag=wx.BOTTOM | wx.RIGHT | wx.LEFT | wx.EXPAND)

    def _init_coll_boxSizerDisplay_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextDisplayAs, 0, border=0,
              flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.radioPixels, 0, border=0, flag=wx.SHRINK)
        parent.AddWindow(self.radioSpheres, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=wx.ALIGN_CENTER_VERTICAL)
        parent.AddWindow(self.staticTextSize, 0, border=0,
              flag=wx.ALIGN_CENTER_VERTICAL)
        parent.AddWindow(self.sliderSize, 0, border=0, flag=0)
        parent.AddWindow(self.checkAutoSize, 0, border=0, flag=0)

    def _init_coll_boxSizerTop_Items(self, parent):
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextViewOption, 0, border=0,
              flag=wx.ALIGN_CENTER_VERTICAL)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.choiceView, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(12, 8), border=0, flag=0)
        parent.AddWindow(self.checkUseSymmetry, 0, border=0,
              flag=wx.ALIGN_CENTER_VERTICAL)
        parent.AddWindow(self.checkHighlightImportant, 0, border=0,
              flag=wx.ALIGN_CENTER_VERTICAL)

    def _init_coll_boxSizerColor_Items(self, parent):
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=wx.ALIGN_CENTER_VERTICAL)
        parent.AddWindow(self.staticTextColor, 0, border=0, flag=wx.ALIGN_CENTER_VERTICAL)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.radioPredicted, 0, border=0, flag=wx.ALIGN_CENTER_VERTICAL)
        parent.AddSpacer(wx.Size(4, 4), border=0, flag=0)
        parent.AddWindow(self.radioMeasured, 0, border=0, flag=wx.ALIGN_CENTER_VERTICAL)
        parent.AddSpacer(wx.Size(12, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextThreshold, 0, border=0, flag=wx.ALIGN_CENTER_VERTICAL)
        parent.AddSpacer(wx.Size(4, 4), border=0, flag=0)
        parent.AddWindow(self.textThreshold, 0, border=0, flag=wx.ALIGN_CENTER_VERTICAL)

    #-------------------------------------------------------------------------------
    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerTop = wx.BoxSizer(orient=wx.HORIZONTAL)
        self.boxSizerColor = wx.BoxSizer(orient=wx.HORIZONTAL)
        self.boxSizerSliceOptions = wx.BoxSizer(orient=wx.HORIZONTAL)
        self.boxSizerDisplay = wx.BoxSizer(orient=wx.HORIZONTAL)

        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerTop_Items(self.boxSizerTop)
        self._init_coll_boxSizerSliceOptions_Items(self.boxSizerSliceOptions)
        self._init_coll_boxSizerDisplay_Items(self.boxSizerDisplay)
        self._init_coll_boxSizerColor_Items(self.boxSizerColor)

        self.SetSizer(self.boxSizerAll)

    #-------------------------------------------------------------------------------
    def _init_ctrls(self, prnt):
        wx.Panel.__init__(self, id=wxID_PANELREFLECTIONSVIEWOPTIONS,
              name=u'PanelReflectionsViewOptions', parent=prnt,
              pos=wx.Point(439, 500), size=wx.Size(606, 269),
              style=wx.TAB_TRAVERSAL)
        self.SetClientSize(wx.Size(606, 269))

        self.panel_to_hold_slice_control = wx.Panel(id=wxID_PANELREFLECTIONSVIEWOPTIONSPANEL_TO_HOLD_SLICE_CONTROL,
              name=u'panel_to_hold_slice_control', parent=self, pos=wx.Point(4,
              89), size=wx.Size(598, 176),
              style=wx.RAISED_BORDER | wx.THICK_FRAME | wx.TAB_TRAVERSAL)
        self.panel_to_hold_slice_control.SetBackgroundColour(wx.Colour(229, 246,
              245))

        self.checkShowSlice = wx.CheckBox(id=wxID_PANELREFLECTIONSVIEWOPTIONSCHECKSHOWSLICE,
              label=u'Show Slice', name=u'checkShowSlice', parent=self,
              pos=wx.Point(4, 67), size=wx.Size(116, 22), style=0)
        self.checkShowSlice.SetValue(True)
        self.checkShowSlice.Bind(wx.EVT_CHECKBOX, self.OnCheckShowSliceCheckbox,
              id=wxID_PANELREFLECTIONSVIEWOPTIONSCHECKSHOWSLICE)

        self.choiceView = wx.Choice(choices=ReflectionsViewChoices,
              id=wxID_PANELREFLECTIONSVIEWOPTIONSCHOICEVIEW, name=u'choiceView',
              parent=self, pos=wx.Point(141, 4), size=wx.Size(260, 29),
              style=0)
        self.choiceView.Bind(wx.EVT_CHOICE, self.OnChangeMaskingSettings)

        self.staticTextViewOption = wx.StaticText(id=wxID_PANELREFLECTIONSVIEWOPTIONSSTATICTEXTVIEWOPTION,
              label=u'Show which peaks:', name=u'staticTextViewOption',
              parent=self, pos=wx.Point(8, 10), size=wx.Size(125, 17), style=0)

        self.checkHighlightImportant = wx.CheckBox(id=wxID_PANELREFLECTIONSVIEWOPTIONSCHECKHIGHLIGHTIMPORTANT,
              label=u'Highlight Important Peaks?',
              name=u'checkHighlightImportant', parent=self, pos=wx.Point(413,
              7), size=wx.Size(215, 22), style=0)
        self.checkHighlightImportant.SetValue(True)
        self.checkHighlightImportant.Hide()

        
        self.checkUseSymmetry = wx.CheckBox(label=u'Use Symmetry?',
              name=u'checkUseSymmetry', parent=self, style=0)
        self.checkUseSymmetry.SetValue(False)
        self.checkUseSymmetry.Bind(wx.EVT_CHECKBOX, self.OnChangeMaskingSettings)
        self.checkUseSymmetry.SetToolTipString("Use the crystal's symmetry to display fewer peaks.")


        self.checkRealtimeSlice = wx.CheckBox(id=wxID_PANELREFLECTIONSVIEWOPTIONSCHECKREALTIMESLICE,
              label=u'Real-time update', name=u'checkRealtimeSlice',
              parent=self, pos=wx.Point(128, 67), size=wx.Size(144, 22),
              style=0)
        self.checkRealtimeSlice.SetValue(False)
        self.checkRealtimeSlice.Bind(wx.EVT_CHECKBOX,
              self.OnCheckRealtimeSliceCheckbox,
              id=wxID_PANELREFLECTIONSVIEWOPTIONSCHECKREALTIMESLICE)

        self.staticTextDisplayAs = wx.StaticText(id=wxID_PANELREFLECTIONSVIEWOPTIONSSTATICTEXTDISPLAYAS,
              label=u'Display as:', name=u'staticTextDisplayAs', parent=self,
              pos=wx.Point(8, 39), size=wx.Size(72, 17), style=0)

        self.radioPixels = wx.RadioButton(id=wxID_PANELREFLECTIONSVIEWOPTIONSRADIOPIXELS,
              label=u'Pixels', name=u'radioPixels', parent=self,
              pos=wx.Point(88, 37), size=wx.Size(85, 22), style=wx.RB_GROUP)
        self.radioPixels.SetValue(True)
        self.radioPixels.Bind(wx.EVT_RADIOBUTTON, self.OnChangeDisplaySettings)

        self.radioSpheres = wx.RadioButton(id=wxID_PANELREFLECTIONSVIEWOPTIONSRADIOSPHERES,
              label=u'Spheres', name=u'radioSpheres', parent=self,
              pos=wx.Point(173, 37), size=wx.Size(85, 22), style=0)
        self.radioSpheres.SetValue(True)
        self.radioSpheres.Bind(wx.EVT_RADIOBUTTON, self.OnChangeDisplaySettings)

        self.staticTextSize = wx.StaticText(id=wxID_PANELREFLECTIONSVIEWOPTIONSSTATICTEXTSIZE,
              label=u'Relative Size: ', name=u'staticTextSize', parent=self,
              pos=wx.Point(266, 39), size=wx.Size(93, 17), style=0)

        self.sliderSize = wx.Slider(id=wxID_PANELREFLECTIONSVIEWOPTIONSSLIDERSIZE,
              maxValue=100, minValue=1, name=u'sliderSize', parent=self,
              pos=wx.Point(359, 37), size=wx.Size(133, 19),
              style=wx.SL_HORIZONTAL, value=0)
        self.sliderSize.Bind(wx.EVT_COMMAND_SCROLL, self.OnChangeDisplaySettings)

        self.checkAutoSize = wx.CheckBox(id=wxID_PANELREFLECTIONSVIEWOPTIONSCHECKAUTOSIZE,
              label=u'Automatic Size', name=u'checkAutoSize',
              parent=self, pos=wx.Point(0, 0),
              size=wx.Size(95, 22), style=0)
        self.checkAutoSize.SetValue(False)
        self.checkAutoSize.Bind(wx.EVT_CHECKBOX, self.OnChangeDisplaySettings)



        self.staticTextColor = wx.StaticText(label=u'Color by:', parent=self, style=0)

        self.radioPredicted = wx.RadioButton(label=u'Predicted', parent=self, style=wx.RB_GROUP)
        self.radioPredicted.Bind(wx.EVT_RADIOBUTTON, self.OnChangeDisplaySettings)

        self.radioMeasured = wx.RadioButton(label=u'Measured', parent=self, style=0)
        self.radioMeasured.Bind(wx.EVT_RADIOBUTTON, self.OnChangeDisplaySettings)
        self.radioPredicted.SetValue(True)

        self.staticTextThreshold = wx.StaticText(label=u'I/sigI threshold', parent=self, style=0)
        self.textThreshold = wx.TextCtrl(value=u'2.0', parent=self, style=wx.TE_PROCESS_ENTER)
        self.textThreshold.Bind(wx.EVT_TEXT_ENTER, self.OntextThreshold)
        self.textThreshold.Bind(wx.EVT_KILL_FOCUS, self.OnChangeMaskingSettings)

        self._init_sizers()

    #-------------------------------------------------------------------------------
    def __init__(self, parent):
        self._init_ctrls(parent)
        
        #Create the view/controller
        self.controller = ReflectionsViewOptionsController(self)

        #Continue by adding the custom slicer control
        self.sliceControl = SliceControl(parent=self.panel_to_hold_slice_control,
                use_slice=False, apply_slice_method=self.controller.apply_slice,
                id=wx.NewId(), name=u'panel_to_hold_slice_control', pos=wx.Point(8, 16),
                size=wx.Size(664, 100))
        #Add a sizer that holds only the sliceControl, put in in the panel_to_hold_slice_control, and expand it
        sizer = wx.BoxSizer()
        sizer.Add(self.sliceControl, 1, wx.EXPAND)
        self.panel_to_hold_slice_control.SetSizer(sizer)

        #Change the UI settings to match the current settings
        self.controller.show_settings()
        self.controller.change_masking_settings()

        #Show the initial data
        self.controller.update_data(None)


    #-------------------------------------------------------------------------------
    def change_displayed_size(self, newsize):
        """Change the pixel size shown in the UI. DO NOT apply the change."""
        self.controller.change_displayed_size(newsize)

    def OnCheckShowSliceCheckbox(self, event):
        self.sliceControl.SetUseSlice( self.checkShowSlice.GetValue() )
        event.Skip()

    def OnCheckRealtimeSliceCheckbox(self, event):
        self.sliceControl.realtime = self.checkRealtimeSlice.GetValue()
        event.Skip()

    def OntextThreshold(self, event):
        self.controller.change_masking_settings()
        event.Skip()

    def OnChangeMaskingSettings(self, event):
        self.controller.change_masking_settings()
        event.Skip()

    def OnChangeDisplaySettings(self, event):
        self.controller.change_display_settings()
        event.Skip()



        
# ===========================================================================================
# ===========================================================================================
# ===========================================================================================

if __name__ == '__main__':
    #Ok, create the instrument
    model.instrument.inst = model.instrument.Instrument("../instruments/TOPAZ_detectors_2010.csv")
    model.instrument.inst.make_qspace()
    #Initialize the instrument and experiment
    model.experiment.exp = model.experiment.Experiment(model.instrument.inst)
    import gui_utils
    (app, pnl) = gui_utils.test_my_gui(PanelReflectionsViewOptions)
    app.frame.SetClientSize(wx.Size(700,500))
    app.MainLoop()


