#Boa:Frame:FrameReflectionInfo
"""Frame that holds the PanelReflectionInfo - info about a single reflection."""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx

#--- GUI Imports ---
from panel_reflection_info import PanelReflectionInfo
import gui_utils

#--- Model Imports ---


#-------------------------------------------------------------------------------
#------ SINGLETON --------------------------------------------------------------
#-------------------------------------------------------------------------------
_instance = None

def create(parent):
    global _instance
    print "FrameReflectionInfo creating a new instance"
    _instance = FrameReflectionInfo(parent)
    _instance.follower = gui_utils.follow_window(parent, _instance, position=gui_utils.FOLLOW_SIDE_TOP)
    return _instance

def get_instance(parent):
    """Returns the singleton instance of this frame."""
    global _instance
    if _instance is None:
        return create(parent)
    else:
        return _instance



[wxID_FRAMEREFLECTIONINFO, wxID_FRAMEREFLECTIONINFOCHECKBOXFOLLOWWINDOW, 
 wxID_FRAMEREFLECTIONINFOSTATICLINE1, wxID_FRAMEREFLECTIONINFOSTATICTEXTHELP, 
] = [wx.NewId() for _init_ctrls in range(4)]

class FrameReflectionInfo(wx.Frame):
    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(6,6), border=0, flag=0)
        parent.AddWindow(self.staticTextHelp, 0, border=0,
              flag=wx.SHRINK | wx.EXPAND)
        parent.AddSpacer(wx.Size(8,8))
        parent.AddWindow(self.staticLine1, 0, border=2,
              flag=wx.BOTTOM | wx.EXPAND | wx.TOP)
        parent.AddWindow(self.checkBoxFollowWindow, 0, border=0, flag=0)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)

        self._init_coll_boxSizerAll_Items(self.boxSizerAll)

        self.SetSizer(self.boxSizerAll)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_FRAMEREFLECTIONINFO,
              name=u'FrameReflectionInfo', parent=prnt, pos=wx.Point(611, 344),
              size=wx.Size(450, 500), style=wx.DEFAULT_FRAME_STYLE,
              title=u'Single Reflection Info')
        self.SetClientSize(wx.Size(450, 500))
        self.Bind(wx.EVT_CLOSE , self.OnFrameClose)

        self.staticTextHelp = wx.StaticText(id=wxID_FRAMEREFLECTIONINFOSTATICTEXTHELP,
              label=u'Type in the HKL of the reflection you want to see, or right-click a peak in the 3D viewer.',
              name=u'staticTextHelp', parent=self, pos=wx.Point(0, 0), style=0)

        self.staticLine1 = wx.StaticLine(id=wxID_FRAMEREFLECTIONINFOSTATICLINE1,
              name='staticLine1', parent=self, pos=wx.Point(0, 50),
              size=wx.Size(400, 2), style=0)

        self.checkBoxFollowWindow = wx.CheckBox(id=wxID_FRAMEREFLECTIONINFOCHECKBOXFOLLOWWINDOW,
              label=u'Follow 3D window', name=u'checkBoxFollowWindow',
              parent=self, pos=wx.Point(0, 54), size=wx.Size(152, 22), style=0)
        self.checkBoxFollowWindow.SetValue(True)
        self.checkBoxFollowWindow.Bind(wx.EVT_CHECKBOX,
              self.OncheckBoxFollowWindowCheck,
              id=wxID_FRAMEREFLECTIONINFOCHECKBOXFOLLOWWINDOW)

        self._init_sizers()

    #-------------------------------------------------------------------------
    def __init__(self, parent, can_follow=True, do_follow=True):
        """Create a new frame.
        Parameters:
            parent: parent frame.
            can_follow: show the "follow window" checkbox.
            do_follow: follow the parent frame when starting.
        """

        self._init_ctrls(parent)

        #Save the parent
        self.parent_frame = parent

        #The checkbox
        if can_follow:
            self.checkBoxFollowWindow.Show()
        else:
            self.checkBoxFollowWindow.Hide()
        self.checkBoxFollowWindow.SetValue(do_follow)
        #Call this to make it start following
        if can_follow and do_follow:
            self.OncheckBoxFollowWindowCheck(None)
            

        #Make the panel and put it in
        self.panel = PanelReflectionInfo(self)
        self.boxSizerAll.Insert(4, self.panel, proportion=1, border=2, flag=wx.EXPAND|wx.SHRINK)


    #-------------------------------------------------------------------------
    def OnFrameClose(self, event):
        #So that the singleton gets re-created if the window is re-opened
        global _instance
        print "frame_reflection_info is being deleted ..."
        if self is _instance:
            _instance = None
        event.Skip()

    #-------------------------------------------------------------------------
    def OncheckBoxFollowWindowCheck(self, event):
        """Change the following window setting."""
        b = self.checkBoxFollowWindow.GetValue()
        if b:
            import frame_qspace_view
            if hasattr(self, 'follower'):
                self.follower.rebind(self.follower.parent, self, position=self.follower.position)
            else:
                self.follower = gui_utils.follow_window(self.parent_frame, self, position=gui_utils.FOLLOW_SIDE_TOP)
        else:
            if hasattr(self, 'follower'):
                self.follower.unbind()
        if not event is None: event.Skip()



if __name__ == "__main__":
    model.instrument.inst = model.instrument.Instrument()
    model.experiment.exp = model.experiment.Experiment(model.instrument.inst)
    model.experiment.exp.initialize_reflections()
    import gui_utils
    (app, pnl) = gui_utils.test_my_gui(FrameReflectionInfo)
#    app.frame.SetClientSize(wx.Size(500, 200))
#    pnl.set_reflection_measurements(None)
#    pnl.set_hkl(1,2,3)
    app.MainLoop()
