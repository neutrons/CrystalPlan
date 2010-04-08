#Boa:Frame:FrameMain
"""FrameMain: main GUI window for the CrystalPlan application.
"""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx.gizmos
import sys

#--- GUI Imports ---
import panel_experiment
import panel_detectors
import panel_goniometer
import panel_add_positions
import panel_try_position
import panel_sample
import panel_startup
import gui_utils
import frame_qspace_view
import frame_reflection_info
import dialog_preferences
import CrystalPlan_version

#--- Model Imports ---
import model


def create(parent):
    return FrameMain(parent)

[wxID_FRAMEMAIN, wxID_FRAMEMAINNOTEBOOK1, wxID_FRAMEMAINPANEL1,
 wxID_FRAMEMAINPANEL2, wxID_FRAMEMAINSPLITTER_MAIN,
 wxID_FRAMEMAINSTATICTEXT1, wxID_FRAMEMAINSTATUSBAR_MAIN,
] = [wx.NewId() for _init_ctrls in range(7)]



class FrameMain(wx.Frame):

    #--------------------------------------------------------------------
    def _init_menuFile(self, parent):
        id = wx.NewId()
        parent.Append(id=id, text=u'Save sample orientations to CSV file...\tCtrl+S', kind=wx.ITEM_NORMAL, help='Make a CSV file containing the list of motor positions.')
        self.Bind(wx.EVT_MENU, self.OnMenuSaveToCSV, id=id)

        id = wx.NewId()
        parent.Append(id=id, text=u'Preferences...', kind=wx.ITEM_NORMAL, help='Change preferences for calculations, display, and others.')
        self.Bind(wx.EVT_MENU, self.OnMenuPreferences, id=id)

        parent.AppendSeparator()

        parent.Append(id=wx.ID_EXIT, text=u'Quit\tCtrl+Q', kind=wx.ITEM_NORMAL, help='Exit the program.')
        self.Bind(wx.EVT_MENU, self.OnMenuQuit, id=wx.ID_EXIT)

    def _init_menuView(self, parent):
        id = wx.NewId()
        parent.Append(id=id, text=u'View Q-Space in 3D\tF2', kind=wx.ITEM_NORMAL, help='Make a CSV file containing the list of motor positions.')
        self.Bind(wx.EVT_MENU, self.OnMenuView3D, id=id)

        id = wx.NewId()
        parent.Append(id=id, text=u'New single reflection info window\tF3', kind=wx.ITEM_NORMAL, help='Open a new window with info for a single HKL reflection.')
        self.Bind(wx.EVT_MENU, self.OnMenuNewReflectionInfoWindow, id=id)


    def _init_menuParameters(self, parent):
        id = wx.NewId()
        parent.Append(id=id, help='', kind=wx.ITEM_NORMAL, text=u'Other...')
        self.Bind(wx.EVT_MENU, self.OnMenu, id=id)


    def _init_menuHelp(self, parent):
        id = wx.NewId()
        parent.Append(id=id, help='', kind=wx.ITEM_NORMAL, text=u'About...')
        self.Bind(wx.EVT_MENU, self.OnMenuAbout, id=id)


    def _init_menus(self):
        self.menuBar1 = wx.MenuBar()
        self.menuBar1.SetHelpText(u'')
        bar = self.menuBar1

        self.menuFile = wx.Menu()
        self._init_menuFile(self.menuFile)
        bar.Append(self.menuFile,"&File")

        self.menuView = wx.Menu()
        self._init_menuView(self.menuView)
        bar.Append(self.menuView,"&View")

#        self.menuParameters = wx.Menu()
#        self._init_menuParameters(self.menuParameters)
#        bar.Append(self.menuParameters, "&Parameters")

        self.menuHelp = wx.Menu()
        self._init_menuHelp(self.menuHelp)
        bar.Append(self.menuHelp,"&Help")


    #--------------------------------------------------------------------
    #--------------- Menu event handlers --------------------------------
    #--------------------------------------------------------------------
    def OnMenuPreferences(self, event):
        res = dialog_preferences.create(self).ShowModal()
        event.Skip()

    def OnMenuSaveToCSV(self,event):
        import gui_utils
        gui_utils.dialog_to_save_experiment_to_CSV(self)
        event.Skip()

    def OnMenuView3D(self, event):
        "Show the 3D view"
        frame_qspace_view.get_instance(self).Show()
        event.Skip()

    def OnMenuQuit(self, event):
        #Make the frame close. Will prompt.
        self.Close()

    def OnMenuNewReflectionInfoWindow(self, event):
        #Make a new frame that can't follow this parent one
        frm = frame_reflection_info.FrameReflectionInfo(self, can_follow=False, do_follow=False)
        frm.Show()
        event.Skip()

    def OnMenuAbout(self, event):
        description = CrystalPlan_version.description
        licence = CrystalPlan_version.license
        info = wx.AboutDialogInfo()
#        info.SetIcon(wx.Icon('icons/hunter.png', wx.BITMAP_TYPE_PNG))
        info.SetName(CrystalPlan_version.package_name)
        info.SetVersion(CrystalPlan_version.version)
        info.SetDescription(description)
        info.SetCopyright(CrystalPlan_version.copyright)
        info.SetWebSite(CrystalPlan_version.url)
        info.SetLicence(licence)
        info.AddDeveloper(CrystalPlan_version.author + " (" + CrystalPlan_version.author_email + ")")
        info.AddDocWriter(CrystalPlan_version.author + " (" + CrystalPlan_version.author_email + ")")

        wx.AboutBox(info)

        event.Skip()

    def OnMenu(self, event):
        event.Skip()
        
    def OnMenu(self, event):
        event.Skip()






    #--------------------------------------------------------------------
    def _init_sizers(self):
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerAll.AddWindow(self.notebook1, 1, border=4, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM)
        self.SetSizer(self.boxSizerAll)


    #--------------------------------------------------------------------
    def _init_ctrls(self, prnt):
        wx.Frame.__init__(self, id=wxID_FRAMEMAIN, name=u'FrameMain',
              parent=prnt, pos=wx.Point(15, 100), size=wx.Size(800, 600),
              style=wx.DEFAULT_FRAME_STYLE,
              title="%s %s - Main Window" % (CrystalPlan_version.package_name, CrystalPlan_version.version) )
        self._init_menus()
        
        self.SetClientSize(wx.Size(800, 700))
        self.SetMenuBar(self.menuBar1)
        self.SetMinSize(wx.Size(100, 100))
        self.SetAutoLayout(True)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.statusBar_main = wx.StatusBar(id=wxID_FRAMEMAINSTATUSBAR_MAIN,
              name=u'statusBar_main', parent=self,
              style=wx.THICK_FRAME | wx.ST_SIZEGRIP)
        self.statusBar_main.SetStatusText(u'Status...')
        self.statusBar_main.SetAutoLayout(True)
        self.SetStatusBar(self.statusBar_main)

        self.notebook1 = wx.Notebook(id=wxID_FRAMEMAINNOTEBOOK1,
              name=u'notebook1', parent=self, pos=wx.Point(0, 0),
              size=wx.Size(573, 629), style=0)
        self.notebook1.SetMinSize(wx.Size(-1, -1))

        self._init_sizers()

    #--------------------------------------------------------------------
    def __init__(self, parent):
        self._init_ctrls(parent)
        #Make the tabs for the notebook
        self.LoadNotebook()
        
        #Subscribe to messages
        model.messages.subscribe(self.OnStatusBarUpdate, model.messages.MSG_UPDATE_MAIN_STATUSBAR)


    #--------------------------------------------------------------------
    def LoadNotebook(self):
        """Add the notebook tabs. """

        self.tab_space = panel_startup.PanelStartup(parent=self.notebook1)
        self.tab_sample = panel_sample.PanelSample(parent=self.notebook1)
        self.tab_goniometer = panel_goniometer.PanelGoniometer(parent=self.notebook1)
        self.tab_experiment = panel_experiment.PanelExperiment(parent=self.notebook1)
        self.tab_add = panel_add_positions.PanelAddPositions(parent=self.notebook1)
        self.tab_try = panel_try_position.PanelTryPosition(parent=self.notebook1)
        self.tab_detectors = panel_detectors.PanelDetectors(parent=self.notebook1)

        def AddPage(tab, title, mac_title="", select=False):
            if gui_utils.is_mac() and not (mac_title==""):
                title = mac_title
            self.notebook1.AddPage(tab, title, select)

        AddPage(self.tab_space, 'Q-Space', 'Q-Space', select=True)
        AddPage(self.tab_detectors, 'Detectors')
        AddPage(self.tab_goniometer, 'Goniometer')
        AddPage(self.tab_sample, 'Sample')
        AddPage(self.tab_try, 'Try an\nOrientation', 'Try Orientation')
        AddPage(self.tab_add, 'Add\nOrientations', 'Add Orientations')
        AddPage(self.tab_experiment, 'Experiment\nPlan', 'Experiment Plan' )
        

    #--------------------------------------------------------------------
    #---------------- other event handlers ------------------------------
    #--------------------------------------------------------------------
    def OnClose(self, event):
        res = wx.MessageDialog(self, "Are you sure you want to quit the program?", "Quit Progam?", wx.YES_NO | wx.YES_DEFAULT).ShowModal()
        if res == wx.ID_YES:
            self.Destroy()
        else:
            event.Veto()

    def OnStatusBarUpdate(self, message):
        """Called when we receive a message that the statusbar needs updating."""
        #print message
        self.statusBar_main.SetStatusText(message.data)        



#--------------------------------------------------------------------
if __name__ == '__main__':
    #Test routine
    model.instrument.inst = model.instrument.Instrument(model.config.cfg.default_detector_filename)
    model.experiment.exp = model.experiment.Experiment(model.instrument.inst)
    (app, frm) = gui_utils.test_my_gui(FrameMain)
    app.MainLoop()
