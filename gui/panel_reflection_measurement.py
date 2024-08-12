#Boa:FramePanel:PanelReflectionMeasurement
"""PanelReflectionMeasurement is a small GUI showing a single measurement for
a single reflection.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx

#--- GUI Imports ---
import detector_plot
import gui_utils
import reflection_placer

#--- Model Imports ---
import model


#================================================================================
#================================================================================
#================================================================================

[wxID_PANELREFLECTIONMEASUREMENT, 
 wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTANGLES, 
 wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTDETECTOR, 
 wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTDETECTORLABEL, 
 wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTMEASUREMENTNUMBER, 
 wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTWL, 
 wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTWLLABEL, 
 wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTX, 
 wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTXLABEL, 
 wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTY, 
 wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTYLABEL, 
] = [wx.NewId() for _init_ctrls in range(11)]


class PanelReflectionMeasurement(wx.Panel):
    """
    PanelReflectionMeasurement is a small GUI showing a single measurement for
    a single reflection.
    """

    MIN_WIDTH = 200
    DEFAULT_WIDTH = 250
    DEFAULT_HEIGHT = 88
    
    def _init_coll_flexGridSizer1_Items(self, parent):
        parent.AddWindow(self.staticTextDetectorLabel, 0, border=0, flag=wx.ALIGN_RIGHT|wx.SHRINK)
        parent.AddWindow(self.staticTextDetector, 0, border=0, flag=wx.SHRINK)
        parent.AddWindow(self.staticTextXLabel, 0, border=0, flag=wx.ALIGN_RIGHT|wx.SHRINK)
        parent.AddWindow(self.staticTextX, 0, border=0, flag=wx.SHRINK)
        parent.AddWindow(self.staticTextWLLabel, 0, border=0, flag=wx.ALIGN_RIGHT|wx.SHRINK)
        parent.AddWindow(self.staticTextWL, 0, border=0, flag=wx.SHRINK)
        parent.AddWindow(self.staticTextYLabel, 0, border=0, flag=wx.ALIGN_RIGHT|wx.SHRINK)
        parent.AddWindow(self.staticTextY, 0, border=0, flag=wx.SHRINK)
        parent.AddWindow(self.staticTextIntegratedLabel, 0, border=0, flag=wx.ALIGN_RIGHT|wx.SHRINK)
        #Spot 9
#        parent.AddWindow(self.staticTextIntegrated, 0, border=0, flag=wx.ALIGN_RIGHT)
#        parent.AddWindow(self.buttonPlace, 0, border=8, flag=wx.EXPAND | wx.RIGHT)
        parent.AddWindow(self.staticTextWidthLabel, 0, border=0, flag=wx.ALIGN_RIGHT|wx.SHRINK)
        parent.AddWindow(self.staticTextWidth, 0, border=0, flag=wx.SHRINK)

    def _init_coll_boxSizerMain_Items(self, parent):
        parent.AddSizer(self.boxSizerTop, 0, border=2, flag=wx.EXPAND| wx.TOP | wx.LEFT)
        parent.AddSizer(self.flexGridSizer1, 0, border=0, flag=wx.EXPAND | wx.BOTTOM)
        parent.AddSpacer(wx.Size(2,2))

    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddSizer(self.boxSizerMain, 0, border=4, flag=wx.EXPAND | wx.BOTTOM| wx.LEFT)
        parent.AddWindow(self.detectorPlot, 1, border=0, flag=wx.EXPAND | wx.BOTTOM)

    def _init_coll_boxSizerTop_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.staticTextMeasurementNumber, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8,8))
        parent.AddWindow(self.staticTextAngles, 0, border=0, flag=wx.SHRINK|wx.ALIGN_CENTER_VERTICAL)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizerAll = wx.BoxSizer(orient=wx.HORIZONTAL)
      
        self.boxSizerTop = wx.BoxSizer(orient=wx.HORIZONTAL)

        self.flexGridSizer1 = wx.FlexGridSizer(cols=4, hgap=1, rows=3, vgap=2)

        self.boxSizerMain = wx.BoxSizer(orient=wx.VERTICAL)

        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerTop_Items(self.boxSizerTop)
        self._init_coll_flexGridSizer1_Items(self.flexGridSizer1)
        self._init_coll_boxSizerMain_Items(self.boxSizerMain)

        self.SetSizer(self.boxSizerAll)
        self.boxSizerAll.Layout()

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_PANELREFLECTIONMEASUREMENT,
              name=u'PanelReflectionMeasurement', parent=prnt, pos=wx.Point(740,
              351), size=wx.Size(self.DEFAULT_WIDTH, 57), style=wx.TAB_TRAVERSAL | wx.SIMPLE_BORDER)
        self.SetClientSize(wx.Size(self.DEFAULT_WIDTH, 57))
        self.SetMinSize(wx.Size(self.DEFAULT_WIDTH, 57))

        self.staticTextAngles = wx.StaticText(id=wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTANGLES,
              label=u'(  0.0,  0.0,  0.0)', name=u'staticTextAngles',
              parent=self, pos=wx.Point(72, 0), style=0)
        self.staticTextAngles.SetFont(wx.Font(10, 76, wx.NORMAL, wx.NORMAL,
              False, u'Courier New'))
        self.staticTextAngles.SetToolTipString(u'Sample orientation angles of this measurement, and HKL of the reflection.')

        self.staticTextMeasurementNumber = wx.StaticText(id=wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTMEASUREMENTNUMBER,
              label=u'#1:', name=u'staticTextMeasurementNumber', parent=self,
              pos=wx.Point(0, 0), style=0)
        self.staticTextMeasurementNumber.SetFont(wx.Font(12, wx.SWISS,
              wx.NORMAL, wx.BOLD, False, u'Sans'))
        self.staticTextMeasurementNumber.SetToolTipString(u'Id of the measurement for this HKL reflection - starting at 0.')

        self.staticTextXLabel = wx.StaticText(id=wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTXLABEL,
              label=u'X:', name=u'staticTextXLabel', parent=self,
              pos=wx.Point(165, 19), style=0)

        self.staticTextYLabel = wx.StaticText(id=wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTYLABEL,
              label=u'Y:', name=u'staticTextYLabel', parent=self,
              pos=wx.Point(165, 39), style=0)

        self.staticTextWLLabel = wx.StaticText(id=wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTWLLABEL,
              label=u'wl:', name=u'staticTextWLLabel', parent=self,
              pos=wx.Point(0, 39), style=0)

        self.staticTextX = wx.StaticText(id=wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTX,
              label=u'0.00 mm', name=u'staticTextX', parent=self,
              pos=wx.Point(217, 19), style=0)
        self.staticTextX.SetToolTipString(u'Horizontal position of the reflection on the detector (0=center)')

        self.staticTextY = wx.StaticText(id=wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTY,
              label=u'0.00 mm', name=u'staticTextY', parent=self,
              pos=wx.Point(217, 39), size=wx.Size(110, 17), style=0)
        self.staticTextY.SetToolTipString(u'Vertical position of the reflection on the detector (0=center)')

        self.staticTextIntegratedLabel = wx.StaticText(label=u'I:',
                parent=self, style=0)
        self.staticTextIntegrated = wx.StaticText(label=u'0.0 ct', parent=self, style=0)
        self.staticTextIntegrated.SetToolTipString(u'Integrated counts under peak.')
        self.staticTextIntegratedLabel.Hide()

        self.staticTextWL = wx.StaticText(id=wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTWL,
              label=u'0.00 ang', name=u'staticTextWL', parent=self,
              pos=wx.Point(52, 39), style=0)
        self.staticTextWL.SetToolTipString(u'Wavelength, in angstroms, at which this reflection was detected.')

        self.staticTextDetectorLabel = wx.StaticText(id=wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTDETECTORLABEL,
              label=u'Det.#:', name=u'staticTextDetectorLabel', parent=self,
              pos=wx.Point(0, 19), style=wx.ST_NO_AUTORESIZE | wx.ALIGN_RIGHT)

        self.staticTextDetector = wx.StaticText(id=wxID_PANELREFLECTIONMEASUREMENTSTATICTEXTDETECTOR,
              label=u'1', name=u'staticTextDetector', parent=self,
              pos=wx.Point(52, 19), size=wx.Size(110, 17), style=0)
        self.staticTextDetector.SetToolTipString(u'Detector number that sees this reflection.')

        self.staticTextWidth = wx.StaticText(label=u'   1.00 mm', name=u'staticTextWidth', parent=self,
              pos=wx.Point(217, 19), style=0)

        #1/2 width unicode symbol
        self.staticTextWidthLabel = wx.StaticText(label=u'\u00BD-width:', name=u'staticTextWidthLabel', parent=self,
              pos=wx.Point(217, 19), style=0)

        self.detectorPlot = detector_plot.DetectorPlot(name=u'detectorPlot', parent=self,
              pos=wx.Point(52, 19), size=wx.Size(5, 5), style=0,
              center_horizontal=False, center_vertical=False, align_right=True )

        self.buttonPlace = wx.Button(label=u'Place...',
              parent=self, pos=wx.Point(128, 62), size=wx.Size(75, 20), style=0)
        self.buttonPlace.Bind(wx.EVT_BUTTON, self.OnButtonPlace)
        self.buttonPlace.SetToolTipString("Open the reflection placer, to move the spot on the detector by changing sample orientation.")
        self.buttonPlace.SetFont(wx.Font(pointSize=8, family=wx.SWISS, weight=wx.NORMAL, style=wx.NORMAL))

        self._init_sizers()

    #---------------------------------------------------------------------------
    def __init__(self, parent):
        self._init_ctrls(parent)
        self.meas = None
        self.refl = None
        #Format string for displaying values
        self.fmt = "%7.2f"
        self.fmt_counts = "%9.1f"
        #Set matching fonts
        for ctl in [self.staticTextX, self.staticTextY, self.staticTextWL,
                    self.staticTextDetector, self.staticTextWidth, self.staticTextIntegrated]:
            ctl.SetFont(wx.Font(11, 76, wx.NORMAL, wx.NORMAL, False, u'Courier New'))
        for ctl in [self.staticTextXLabel, self.staticTextYLabel, self.staticTextWLLabel,
                    self.staticTextDetectorLabel, self.staticTextWidthLabel, self.staticTextIntegratedLabel]:
            ctl.SetFont(wx.Font(11, 76, wx.NORMAL, wx.NORMAL, False, u'Courier New'))

    #---------------------------------------------------------------------------
    def set_measurement(self, refl, meas):
        """Make the panel display the given ReflectionMeasurement object 'meas'"""
        self.refl = refl
        #@type meas: ReflectionMeasurement
        self.meas = meas
        #Also for the detector plot
        self.detectorPlot.set_measurement(meas)
        
        if meas is None:
            self.staticTextAngles.SetLabel("None")
            self.staticTextWL.SetLabel("None")
            self.staticTextX.SetLabel("None")
            self.staticTextY.SetLabel("None")
            self.staticTextDetector.SetLabel("None")
            self.staticTextWidth.SetLabel("None")
            self.staticTextMeasurementNumber.SetLabel("---")
        else:
            fmt = self.fmt
            hkl_str = "%d,%d,%d" % refl.hkl
            self.staticTextAngles.SetLabel(meas.make_sample_orientation_string() + " as HKL %s" % hkl_str )

            det_name = "None"
            try:
                det_name = model.instrument.inst.detectors[meas.detector_num].name
            except:
                pass

            self.staticTextDetector.SetLabel(" %s" % (det_name))
            self.staticTextWL.SetLabel((fmt % meas.wavelength) + u" \u212B") #Angstrom symbol
            self.staticTextX.SetLabel((fmt % meas.horizontal) + " mm")
            self.staticTextY.SetLabel((fmt % meas.vertical) + " mm")
            self.staticTextMeasurementNumber.SetLabel("#%d:" % meas.measurement_num)

            #Remove these windows, if they are in there
            try:
                self.flexGridSizer1.Remove(self.staticTextIntegrated)
            except:
                pass
            try:
                self.flexGridSizer1.Remove(self.buttonPlace)
            except:
                pass
      
            if hasattr(meas, "integrated"):
                #Real measurement
                self.staticTextWidthLabel.SetLabel(" SigI:")
                self.staticTextWidth.SetLabel((self.fmt_counts % meas.sigI))
                self.staticTextWidth.SetToolTipString(u'Sigma I of the integrated peak intensity')
                self.staticTextIntegrated.SetLabel((self.fmt_counts % meas.integrated))
                self.flexGridSizer1.Detach(self.buttonPlace)
                self.flexGridSizer1.Detach(self.staticTextIntegrated)
                self.flexGridSizer1.InsertWindow(9, self.staticTextIntegrated, 0, border=0, flag=wx.SHRINK)
                self.staticTextIntegratedLabel.Show()
                self.staticTextIntegrated.Show()
                self.buttonPlace.Hide()
            else:
                # Predicted - normal mode
                self.staticTextWidthLabel.SetLabel("Width:")
                self.staticTextWidth.SetLabel((fmt % meas.peak_width) + " mm")
                self.staticTextWidth.SetToolTipString(u'Half-width of the peak on the detector.')
                self.flexGridSizer1.Detach(self.buttonPlace)
                self.flexGridSizer1.Detach(self.staticTextIntegrated)
                self.flexGridSizer1.InsertWindow(9, self.buttonPlace, 0, border=8, flag=wx.EXPAND | wx.RIGHT)
                # Don't show the place button for 4-circle
                self.buttonPlace.Show( not gui_utils.fourcircle_mode() )
                self.staticTextIntegratedLabel.Hide()
                self.staticTextIntegrated.Hide()



    def OnButtonPlace(self, event):
        self.last_placer_frame = reflection_placer.show_placer_frame(self, self.refl, self.meas)
        event.Skip()


if __name__ == "__main__":
    import gui_utils
    model.instrument.inst = model.instrument.InstrumentFourCircle()
    (app, pnl) = gui_utils.test_my_gui(PanelReflectionMeasurement)
    ref = model.reflections.Reflection( (1,2,3), (2., 3., 4.))
    ref.measurements = [ (0,0,0,0,0,0)]
    #@type meas: ReflectionMeasurement
    meas = model.reflections.ReflectionMeasurement(ref, 0)
    meas.wavelength=1.23
    meas.horizontal=12.34
    meas.vertical=-45.2
    pnl.set_measurement( ref, meas )
    #Start the fun!
    app.MainLoop()

