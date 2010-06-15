#Boa:FramePanel:ValueSlider
"""Fairly generic control composed of a label, a slider and a textbox.
Both control the same number.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import wx.lib.newevent


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
[wxID_VALUESLIDER, wxID_VALUESLIDERSLIDERVALUE, 
 wxID_VALUESLIDERSTATICTEXTLABEL, wxID_VALUESLIDERTEXTVALUE, 
] = [wx.NewId() for _init_ctrls in range(4)]


#At the end of movement
ValueSliderChanged, EVT_VALUE_SLIDER_CHANGED = wx.lib.newevent.NewEvent()
#During a scroll
ValueSliderChanging, EVT_VALUE_SLIDER_CHANGING = wx.lib.newevent.NewEvent()

#--------------------------------------------------------------------------------
class ValueSlider(wx.Panel):
    """Fairly generic control composed of a label, a slider and a textbox.
    Both control the same number."""
    Min = 0.0
    Max = 100.0
    Value = 0.0
    #Internal values
    _insideSetvalue = False
    
    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddSizer(self.boxSizerTop, 0, border=0, flag=wx.EXPAND)
        parent.AddWindow(self.sliderValue, 0, border=0, flag=wx.EXPAND)

    def _init_coll_boxSizerTop_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.staticTextLabel, 1, border=0,
              flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.textValue, 0, border=0, flag=0)

    def SetLabel(self, text):
        """Set the text of the label."""
        self.staticTextLabel.SetLabel(text)
        
    def SetMin(self, value):
        """Sets the minimum value allowed in the control."""
        self.Min = value
        self.sliderValue.SetMin( int(value*self.float_convert) )
        self.sliderValue.SetPageSize(5)
        self._check_value()
        
    def SetMax(self, value):
        """Sets the maximum value allowed in the control."""
        self.Max = value
        self.sliderValue.SetMax( int(value*self.float_convert) )
        self.sliderValue.SetPageSize(5)
        self._check_value()
        
    def SetValue(self, value, change_text=True):
        """Sets the value shown in the control. Text gets updated unless otherwise specified."""
        self._insideSetvalue = True
        self.Value = value
        if self.Value < self.Min: self.Value = self.Min
        if self.Value > self.Max: self.Value = self.Max
        self.sliderValue.SetValue( int(self.Value * self.float_convert) )
        if change_text:
            self.textValue.SetValue( str(self.Value) )
        self._insideSetvalue = False
        
    def _check_value(self):
        """Makes sure that the values entered are within the bounds."""
        if self.Value < self.Min:
            self.SetValue(self.Min)
        if self.Value > self.Max:
            self.SetValue(self.Max)
        
    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)

        self.boxSizerTop = wx.BoxSizer(orient=wx.HORIZONTAL)

        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerTop_Items(self.boxSizerTop)

        self.SetSizer(self.boxSizerAll)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_VALUESLIDER, name=u'ValueSlider',
              parent=prnt, pos=wx.Point(639, 289), size=wx.Size(160, 61),
              style=wx.TAB_TRAVERSAL)
        self.SetClientSize(wx.Size(160, 61))

        self.staticTextLabel = wx.StaticText(id=wxID_VALUESLIDERSTATICTEXTLABEL,
              label=u'Label:', name=u'staticTextLabel', parent=self,
              pos=wx.Point(0, 0), size=wx.Size(128, 27), style=0)

        self.textValue = wx.TextCtrl(id=wxID_VALUESLIDERTEXTVALUE,
              name=u'textValue', parent=self, pos=wx.Point(80, 0),
              size=wx.Size(80, 27), style=wx.TE_PROCESS_ENTER, value=u'0')
        self.textValue.Bind(wx.EVT_TEXT, self.OnTextValueText)
        self.textValue.Bind(wx.EVT_TEXT_ENTER, self.OnTextEnter) #Don't forget the wx.TE_PROCESS_ENTER otherwise it doesn't work.
        self.textValue.Bind(wx.EVT_KILL_FOCUS, self.OnTextEnter)
        self.textValue.Bind(wx.EVT_KEY_UP, self.OnTextValueKeyUp)

        self.sliderValue = wx.Slider(id=wxID_VALUESLIDERSLIDERVALUE,
              maxValue=100, minValue=0, name=u'sliderValue', parent=self,
              pos=wx.Point(0, 27), size=wx.Size(200, 19),
              style=wx.SL_HORIZONTAL, value=0)
        self.sliderValue.Bind(wx.EVT_SCROLL, self.OnSliderValueScroll)
        self.sliderValue.Bind(wx.EVT_SCROLL_LINEUP, self.OnSliderLineOrPage)
        self.sliderValue.Bind(wx.EVT_SCROLL_LINEDOWN, self.OnSliderLineOrPage)
        self.sliderValue.Bind(wx.EVT_SCROLL_PAGEUP, self.OnSliderLineOrPage)
        self.sliderValue.Bind(wx.EVT_SCROLL_PAGEDOWN, self.OnSliderLineOrPage)
        self.sliderValue.Bind(wx.EVT_SCROLL_THUMBRELEASE, self.OnSliderValueScrollThumbrelease)

        self._init_sizers()

    def __init__(self, parent, floats=0):
        self._init_ctrls(parent)
        self.floats=floats
        self.float_convert = 10.0**floats
        self.SetMax(100)
        self.SetMin(0)

    #-------------- CUSTOM EVENTS --------------------------
    def SendScrollEvent(self):
        """Sends out the scroll event (while typing or moving the slider)."""
        scrollEvent = ValueSliderChanging()
        self.GetEventHandler().ProcessEvent(scrollEvent)

    def SendScrollEndEvent(self):
        """Sends out the scroll end event (when releasing the slider or pressing enter)."""
        scrollEndEvent = ValueSliderChanged()
        self.GetEventHandler().ProcessEvent(scrollEndEvent)

    #-------------- Slider Event Handlers ------------------
    def OnSliderValueScroll(self, event):
        self.SetValue( self.sliderValue.GetValue()/self.float_convert )
        self.SendScrollEvent()
        event.Skip()

    def OnSliderValueScrollThumbrelease(self, event):
        self.SetValue( self.sliderValue.GetValue()/self.float_convert )
        self.SendScrollEndEvent()
        event.Skip()

    def OnSliderLineOrPage(self, event):
        "When pressing line up/down or page up/down"
        self.SetValue( self.sliderValue.GetValue()/self.float_convert )
        self.SendScrollEndEvent()
        event.Skip()


    #-------------- Textbox Event Handlers ------------------
    def OnTextValueKeyUp(self, event):
        """Called when pevery key is released."""
        if event.GetKeyCode() == wx.WXK_RETURN:
            self.SetValue ( self.Value, change_text=True )
            self.SendScrollEndEvent()
        event.Skip()
        
    def OnTextEnter(self, event):
        """Called when pressing enter or losing focus."""
        self.SetValue ( self.Value, change_text=True )
        self.SendScrollEndEvent()
        event.Skip()

    def OnTextValueText(self, event):
        if not self._insideSetvalue:
            try:
                val = float( self.textValue.GetValue() )
            except:
                val = 0
            self.SetValue ( val, change_text=False )
            self.SendScrollEvent()
        event.Skip()






#===========================================================================
def on_slider(event):
    global slid
    print "Scroll End: ", slid.Value

def on_slider_scroll(event):
    global slid
    print "Scrolling ", slid.Value

if __name__ == '__main__':
    import gui_utils
    global slid
    (app, slid) = gui_utils.test_my_gui(ValueSlider, floats=2)
    slid.Bind(EVT_VALUE_SLIDER_CHANGED, on_slider)
    slid.Bind(EVT_VALUE_SLIDER_CHANGING, on_slider_scroll)
    app.MainLoop()
#    #Test routine
#    from wxPython.wx import *
#
#    class MyApp(wxApp):
#        def OnInit(self):
#            frame = wxFrame(NULL, -1, "Hello from wxPython")
#            frame.Show(true)
#            self.vs = ValueSlider(parent=frame, id=wx.NewId(), pos=wx.Point(0,0), size=wx.Size(200,60), style=0, name="slider")
#            self.vs.Bind(EVT_VALUE_SLIDER_CHANGED, self.Changed2)
#            self.vs.SetLabel("Phi:")
#            self.SetTopWindow(frame)
#            return true
#
#        def Changed2(self, event):
#            print "Done", self.vs.Value
#
#    app = MyApp(0)
#    app.MainLoop()

