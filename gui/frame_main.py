"""FrameMain: main GUI window for the CrystalPlan application.
"""
import os.path
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import wx.gizmos
import sys
import os

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
import display_thread
import frame_optimizer
import detector_plot

#--- Model Imports ---
import model


[wxID_FRAMEMAIN, wxID_FRAMEMAINnotebook, wxID_FRAMEMAINPANEL1,
 wxID_FRAMEMAINPANEL2, wxID_FRAMEMAINSPLITTER_MAIN,
 wxID_FRAMEMAINSTATICTEXT1, wxID_FRAMEMAINSTATUSBAR_MAIN,
] = [wx.NewId() for _init_ctrls in range(7)]


#
class FrameMain(wx.Frame):

    #--------------------------------------------------------------------
    def _init_menuFile(self, parent):
        id = wx.NewId()
        parent.Append(id=id, text=u'Save experiment to file...\tCtrl+S', kind=wx.ITEM_NORMAL,
                    help='Save the experiment to a .exp file so that it can be re-loaded later.')
        self.Bind(wx.EVT_MENU, self.OnMenuSave, id=id)

        id = wx.NewId()
        parent.Append(id=id, text=u'Load experiment from file...\tCtrl+L', kind=wx.ITEM_NORMAL,
                    help='Load a .exp file.')
        self.Bind(wx.EVT_MENU, self.OnMenuLoad, id=id)

        parent.AppendSeparator()

        if gui_utils.fourcircle_mode():
            # ---------- HFIR-specific loading menus --------------------

            id = wx.NewId()
            parent.Append(id=id, text=u'Load a HFIR .int file...\tCtrl+I', kind=wx.ITEM_NORMAL,
                        help='Load a peaks file from HFIR software to compare predicted and real peaks.')
            self.Bind(wx.EVT_MENU, self.OnMenuLoadIntegrateHFIR, id=id)

            id = wx.NewId()
            parent.Append(id=id, text=u'Load a HFIR UB matrix and lattice parameters file...\tCtrl+U', kind=wx.ITEM_NORMAL,
                        help='Load a UB matrix file made by HFIR software, and a corresponding lattice parameters file.')
            self.Bind(wx.EVT_MENU, self.OnMenuLoadHFIRUB, id=id)

        else:
            # ---------- ISAW loading menus --------------------
            id = wx.NewId()
            parent.Append(id=id, text=u'Load an older ISAW .integrate or .peaks file with sequential det. numbers', kind=wx.ITEM_NORMAL,
                        help='Load a peaks file from ISAW to compare predicted and real peaks. Use this menu for older files (made with ISAW before ~April 2011).')
            self.Bind(wx.EVT_MENU, self.OnMenuLoadIntegrateOld, id=id)

            id = wx.NewId()
            parent.Append(id=id, text=u'Load an ISAW .integrate or .peaks file...\tCtrl+I', kind=wx.ITEM_NORMAL,
                        help='Load a peaks file from ISAW to compare predicted and real peaks.')
            self.Bind(wx.EVT_MENU, self.OnMenuLoadIntegrateNew, id=id)

            id = wx.NewId()
            parent.Append(id=id, text=u'Load an ISAW UB matrix (.mat) file...\tCtrl+U', kind=wx.ITEM_NORMAL,
                        help='Load a UB matrix file made by ISAW (goniometer-corrected; not by ISAWev).')
            self.Bind(wx.EVT_MENU, self.OnMenuLoadUB, id=id)


            id = wx.NewId()
            parent.Append(id=id, text=u'Load a Lauegen .ldm UB matrix file...\tCtrl+M', kind=wx.ITEM_NORMAL,
                        help='Load a Lauegen .ldm file containing the crystal orientation for the IMAGINE instrument.')
            self.Bind(wx.EVT_MENU, self.OnMenuLoadLDM, id=id)


        parent.AppendSeparator()

        id = wx.NewId()
        parent.Append(id=id, text=u'Save sample orientations to CSV file...\tCtrl+D', kind=wx.ITEM_NORMAL,
                    help='Make a CSV file containing the list of motor positions.')
        self.Bind(wx.EVT_MENU, self.OnMenuSaveToCSV, id=id)

        id = wx.NewId()
        parent.Append(id=id, text=u'Preferences...', kind=wx.ITEM_NORMAL, help='Change preferences for calculations, display, and others.')
        self.Bind(wx.EVT_MENU, self.OnMenuPreferences, id=id)

        parent.AppendSeparator()

        parent.Append(id=wx.ID_EXIT, text=u'Quit\tCtrl+Q', kind=wx.ITEM_NORMAL, help='Exit the program.')
        self.Bind(wx.EVT_MENU, self.OnMenuQuit, id=wx.ID_EXIT)

    # -------------------------------------------------------------------------
    def _init_menuView(self, parent):
        id = wx.NewId()
        parent.Append(id=id, text=u'View Q-Space in 3D\tF2', kind=wx.ITEM_NORMAL, help='Make a CSV file containing the list of motor positions.')
        self.Bind(wx.EVT_MENU, self.OnMenuView3D, id=id)

        id = wx.NewId()
        parent.Append(id=id, text=u'New single reflection info window\tF3', kind=wx.ITEM_NORMAL, help='Open a new window with info for a single HKL reflection.')
        self.Bind(wx.EVT_MENU, self.OnMenuNewReflectionInfoWindow, id=id)


    # -------------------------------------------------------------------------
    def _init_menuParameters(self, parent):
        id = wx.NewId()
        parent.Append(id=id, help='', kind=wx.ITEM_NORMAL, text=u'Other...')
        self.Bind(wx.EVT_MENU, self.OnMenu, id=id)


    # -------------------------------------------------------------------------
    def _init_menuTools(self, parent):

        if not gui_utils.fourcircle_mode():
            id = wx.NewId()
            parent.Append(id=id, help='', kind=wx.ITEM_NORMAL, text=u'Automatic Coverage Optimizer...\tCtrl+O')
            self.Bind(wx.EVT_MENU, self.OnMenuOptimizePositions, id=id)

        id = wx.NewId()
        parent.Append(id=id, help='', kind=wx.ITEM_NORMAL, text=u'Compare measured to predicted peak positions...')
        self.Bind(wx.EVT_MENU, self.OnMenuComparePeaks, id=id)

        id = wx.NewId()
        parent.Append(id=id, help='', kind=wx.ITEM_NORMAL, text=u'Find angles for all HKL.')
        self.Bind(wx.EVT_MENU, self.OnMenuFourCircleAllHKL, id=id)

        id = wx.NewId()
        parent.Append(id=id, help='', kind=wx.ITEM_NORMAL, text=u'Simple Laue Plots')
        self.Bind(wx.EVT_MENU, self.OnMenuLauePlot, id=id)


    def _init_menuHelp(self, parent):
        id = wx.NewId()
        parent.Append(id=id, help='', kind=wx.ITEM_NORMAL, text=u'Open User Guide in WebBrowser\tF1')
        self.Bind(wx.EVT_MENU, self.OnMenuUserGuide, id=id)

        id = wx.NewId()
        parent.Append(id=id, help='', kind=wx.ITEM_NORMAL, text=u'About %s...' % CrystalPlan_version.package_name)
        self.Bind(wx.EVT_MENU, self.OnMenuAbout, id=id)

#        parent.AppendSeparator()
#
#        id = wx.NewId()
#        parent.Append(id=id, help='', kind=wx.ITEM_NORMAL, text=u'Generate User Guide (ADVANCED)\tCtrl+H')
#        self.Bind(wx.EVT_MENU, self.OnMenuGenerateUserGuide, id=id)


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

        self.menuTools = wx.Menu()
        self._init_menuTools(self.menuTools)
        bar.Append(self.menuTools, "&Tools")

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
        gui_utils.dialog_to_save_experiment_to_CSV(self)
        event.Skip()

    def OnMenuSave(self,event):
        gui_utils.save_experiment_file_dialog(self)
        event.Skip()

    def OnMenuLoadIntegrateNew(self,event):
        gui_utils.load_integrate_file_dialog(self, sequential_detector_numbers=False)
        event.Skip()

    def OnMenuLoadIntegrateOld(self,event):
        gui_utils.load_integrate_file_dialog(self, sequential_detector_numbers=True)
        event.Skip()

    def OnMenuLoadIntegrateHFIR(self, event):
        gui_utils.load_HFIR_int_file_dialog(self)
        event.Skip()


    def OnMenuLoadUB(self,event):
        self.load_ubmatrix_file_dialog(self)
        event.Skip()

    def OnMenuLoadHFIRUB(self, event):
        self.load_HFIR_ubmatrix_file_dialog(self)
        event.Skip()
        
    def OnMenuLoadLDM(self,event):
        self.load_ldm_file_dialog(self)
        self.RefreshAll()
        #This message will make the experiment lists update, etc.
        model.messages.send_message(model.messages.MSG_GONIOMETER_CHANGED, "")
        event.Skip()
        


    def load_HFIR_ubmatrix_file_dialog(self, parent):
        """Opens a dialog asking the user where to load the ubmatrix file."""
        filters = 'HFIR UB matrix file (*.dat)|*.dat|All files (*)|*|'
        (path, filename) = os.path.split(self.last_ubmatrix_path)
        dialog = wx.FileDialog ( parent, defaultFile=filename, defaultDir=path, message='Load a HFIR UB Matrix file', wildcard=filters, style=wx.OPEN )
        if dialog.ShowModal() == wx.ID_OK:
            filename = dialog.GetPath()
            self.last_ubmatrix_path = filename
            dialog.Destroy()
        else:
            #'Nothing was selected.
            dialog.Destroy()
            return None

        (path, load_filename) = os.path.split(self.last_lattice_path)
        if self.last_lattice_path == "":
            path = os.path.split(self.last_ubmatrix_path)[0]
            load_filename = ''
        filters = 'HFIR lattice parameters file (*.dat)|*.dat|All files (*)|*|'
        dialog = wx.FileDialog ( parent, defaultFile=load_filename, defaultDir=path, message='Load a HFIR lattice parameters file', wildcard=filters, style=wx.OPEN )
        if dialog.ShowModal() == wx.ID_OK:
            lattice_filename = dialog.GetPath()
            self.last_lattice_path = lattice_filename
            dialog.Destroy()
        else:
            #'Nothing was selected.
            dialog.Destroy()
            return None

        #The old U matrix, before messing with it.
        old_U = model.experiment.exp.crystal.get_u_matrix()

        print filename
        print lattice_filename

        #Load the file with no goniometer correction
        model.experiment.exp.crystal.read_HFIR_ubmatrix_file(filename, lattice_filename)

        #Now this handles updating all the gui etc.
        self.tab_sample.OnReturningFromEditCrystal(old_U)

        

    def load_ubmatrix_file_dialog(self, parent):
        """Opens a dialog asking the user where to load the ubmatrix file."""
        filters = 'ISAW UB matrix file (*.mat)|*.mat|All files (*)|*|'
        (path, filename) = os.path.split(self.last_ubmatrix_path)
        dialog = wx.FileDialog ( parent, defaultFile=filename, defaultDir=path, message='Load an ISAW (goniometer-corrected) .mat file', wildcard=filters, style=wx.OPEN )
        if dialog.ShowModal() == wx.ID_OK:
            filename = dialog.GetPath()
            self.last_ubmatrix_path = filename
            dialog.Destroy()
        else:
            #'Nothing was selected.
            dialog.Destroy()
            return None

        #The old U matrix, before messing with it.
        old_U = model.experiment.exp.crystal.get_u_matrix()

        #Load the file with no goniometer correction
        model.experiment.exp.crystal.read_ISAW_ubmatrix_file(filename, angles=[0,0,0])
        #TODO: Check if ISAW matrix file has line saying NOT GONIOMETER CORRECTED

        #Now this handles updating all the gui etc.
        self.tab_sample.OnReturningFromEditCrystal(old_U)

    def load_ldm_file_dialog(self, parent):
        """Opens a dialog asking the user where to load the .ldm file."""
        filters = 'Lauegen .ldm file (*.ldm)|*.ldm|All files (*)|*|'
        (path, filename) = os.path.split(self.last_ldm_path)
        dialog = wx.FileDialog ( parent, defaultFile=filename, defaultDir=path, message='Load an Lauegen .ldm file', wildcard=filters, style=wx.OPEN )
        if dialog.ShowModal() == wx.ID_OK:
            filename = dialog.GetPath()
            self.last_ldm_path = filename
            dialog.Destroy()
        else:
            #'Nothing was selected.
            dialog.Destroy()
            return None
        model.experiment.exp.inst = model.instrument.Instrument("instruments/IMAGINE_detectors.csv")
        model.experiment.exp.inst.set_goniometer(model.goniometer.ImagineGoniometer())

        #The old U matrix, before messing with it.
        old_U = model.experiment.exp.crystal.get_u_matrix()

        #Load the file 
        (d_min,wl_min,wl_max) = model.experiment.exp.crystal.read_LDM_file(filename)
        if d_min > 0:
            model.experiment.exp.inst.d_min = d_min
            import numpy as np
            model.experiment.exp.inst.q_resolution = 2*np.pi / d_min
            model.experiment.exp.inst.wl_min = wl_min
            model.experiment.exp.inst.wl_max = wl_max
            model.experiment.exp.inst.make_qspace()

        #Now this handles updating all the gui etc.
        self.tab_sample.OnReturningFromEditCrystal(old_U)
        #Now we need to fix a lot of stuff
        model.instrument.inst = model.experiment.exp.inst
        #This hopefully redraws everything
        display_thread.handle_change_of_qspace()
        #Make sure we select it all, by default.
        display_thread.select_position_coverage(poscov_list=model.instrument.inst.positions, update_gui=True)

    def OnMenuLoad(self,event):
        if not gui_utils.load_experiment_file_dialog(self) is None:
            self.RefreshAll()
            #This message will make the experiment lists update, etc.
            model.messages.send_message(model.messages.MSG_GONIOMETER_CHANGED, "")
        event.Skip()

    def OnMenuView3D(self, event):
        "Show the 3D view"
        frame_qspace_view.get_instance(self).Show()
        #Bring it to front
        frame_qspace_view.get_instance(self).Raise()
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
        info = wx.AboutDialogInfo()
        info.SetName(CrystalPlan_version.package_name)
        info.SetVersion(CrystalPlan_version.version)
        info.SetDescription(CrystalPlan_version.description)
        info.SetCopyright(CrystalPlan_version.copyright)
        info.AddDeveloper(CrystalPlan_version.author + " (" + CrystalPlan_version.author_email + ")")
        info.AddDocWriter(CrystalPlan_version.author + " (" + CrystalPlan_version.author_email + ")")
        info.AddArtist("Icons taken from the Crystal Project,\nat http://www.everaldo.com/crystal/, \ndistributed under the LGPL; \nmodified and assembled by Janik Zikovsky")
        
        if not gui_utils.is_mac():
            #Some of these are not natively mac-supported, not including them makes it look better on mac
            icon_file = os.path.join(os.path.dirname(__file__), CrystalPlan_version.icon_file)
            info.SetIcon(wx.Icon( icon_file, wx.BITMAP_TYPE_PNG))
            info.SetLicence(CrystalPlan_version.license)
            info.SetWebSite(CrystalPlan_version.url)

        wx.AboutBox(info)

        event.Skip()

    def OnMenuUserGuide(self, event):
        """Open the user guide in a browser."""
        filename = "../docs/user_guide.html"
        absolute_file = os.path.abspath( os.path.split(__file__)[0] + "/" + filename)
        if gui_utils.is_mac():
            #Start browser on mac
            command = ['open', '"%s"' % absolute_file]
        else:
            #Start firefox
            command = ['firefox', '"%s"' % absolute_file]
        #Open in background?

        import subprocess
        #result = subprocess.Popen(command)
        os.system(" ".join(command + [" &"]))


        #if result != 0:
        #    absolute_file = os.path.abspath( os.path.split(__file__)[0] + "/" + filename)
        #    wx.MessageDialog(self, 'Sorry! There was an error opening the user guide in the webbrowser. You can find it at:\n\n%s\n\n(You can copy/paste the file path above)' % absolute_file,
        #                    'Error Opening User Guide', wx.OK|wx.ICON_ERROR).ShowModal()
            
        event.Skip()

    def OnMenuGenerateUserGuide(self, event):
        import doc_maker
        import doc_maker.screenshots
        import doc_maker.user_guide

        #Make the user guide screenshots
        if hasattr(self, 'user_guide_thread'):
            print "The thread has already started!"
        else:
            self.user_guide_thread = doc_maker.user_guide.generate_user_guide(self, frame_qspace_view.get_instance(self))
        event.Skip()


    def OnMenuOptimizePositions(self, event):
        frm = frame_optimizer.get_instance(parent=self)
        frm.Show()
        frm.Raise()
        event.Skip()

    def OnMenuFourCircleAllHKL(self, event):
        max = len( model.experiment.exp.reflections)
        prog_dlg = wx.ProgressDialog( "Calculating goniometer angles for all HKL.",   "Calculation progress:",
            max, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_REMAINING_TIME | wx.PD_AUTO_HIDE)
        #Make it wider
        prog_dlg.SetSize((500, prog_dlg.GetSize()[1]))
        # This performs the calculation
        model.experiment.exp.fourcircle_measure_all_reflections(prog_dlg)
        # Update the main GUI
        self.RefreshAll()
        # Remove the progress bar
        prog_dlg.Destroy()

        event.Skip()

    def OnMenuComparePeaks(self, event):
        #First, do the calculation.
        offsets = model.tools.calculate_peak_offsets()
        if len(offsets) == 0:
            wx.MessageDialog(self, "No peaks were found to have any equivalent predicted peaks. Make sure you have loaded a .integrate file and that your list of sample orientations matches the one in the .integrate file.").ShowModal()
            return
        #Prompt for a file
        global last_csv_path
        (path, filename) = os.path.split( os.path.expanduser("~") + "/peak_offsets" )
        dialog = wx.FileDialog ( self, defaultFile=filename, defaultDir=path, message='Enter the base file to save to (no extension)', wildcard='*', style=wx.SAVE )
        if dialog.ShowModal() == wx.ID_OK:
            filename = dialog.GetPath()
            dialog.Destroy()
            #Do the figure and csv file
            model.tools.save_offsets_to_csv(offsets, filename + '.csv')
            model.tools.plot_peak_offsets(offsets, filename, doshow=False)
            wx.MessageDialog(self, "Offsets were saved to '%s*.pdf' and '%s.csv'." % (filename, filename)).ShowModal()
            return
        else:
            #'Nothing was selected.
            dialog.Destroy()
            return
        
    def make_Laue_plot(self, det, detnum, parent):
        """ Create a simple Laue plot.
        Parameters
             det :: detector object
             detnum :: index of the detector
             parent :: parent wxWindow
        """
#        frame = wx.Window(parent)
#        boxSizer = wx.BoxSizer(orient=wx.VERTICAL)
#        frame.SetSizer(boxSizer)
#        
        # Now show the detector
        plot = detector_plot.DetectorPlot(parent)
        plot.set_detector(det)
        plot.SetToolTip(wx.ToolTip("Detector %s" % det.name))
        
        # Collect all the measurements
        measures = []
        for ref in model.experiment.exp.reflections:
            n = 0
            for meas in ref.measurements:
                # each tuple holds  (poscov_id, detector_num, horizontal, vertical, wavelength, distance)
                if meas[1] == detnum:
                    measures.append( model.reflections.ReflectionMeasurement(ref, n))
        # Set them all for this detector
        plot.set_measurements(measures, det)
                
#        # Do layout        
#        label = wx.StaticText(parent, -1, "Detector %s" % det.name)
#        boxSizer.Add(label,1, flag=wx.EXPAND)
#        boxSizer.Add(plot,1, flag=wx.EXPAND)
#        return frame
    
        return plot

    def OnMenuLauePlot(self, event):
        """ Do a laue plot for each detector """
        #Simple frame
        frame = wx.Frame(None, title='Laue Plot')
        boxSizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        frame.SetSizer(boxSizer)
        
        # Make a plot for each detector
        detnum = 0
        for det in model.instrument.inst.detectors:
            plot = self.make_Laue_plot(det, detnum, frame)
            #Make it resize
            boxSizer.Add(plot, 1, flag=wx.EXPAND)
            detnum += 1
            
        frame.Show()
        event.Skip()

    def OnMenu(self, event):
        event.Skip()






    #--------------------------------------------------------------------
    def _init_sizers(self):
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerAll.AddWindow(self.notebook, 1, border=4, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM)
        self.SetSizer(self.boxSizerAll)


    #--------------------------------------------------------------------
    def _init_ctrls(self, prnt):
        wx.Frame.__init__(self, id=wxID_FRAMEMAIN, name=u'FrameMain',
              parent=prnt, pos=wx.Point(0, 0), size=wx.Size(800, 630),
              style=wx.DEFAULT_FRAME_STYLE,
              title="%s %s - Main Window" % (CrystalPlan_version.package_name, CrystalPlan_version.version) )
        self._init_menus()
        
        self.SetClientSize(wx.Size(850, 630))
        self.SetMenuBar(self.menuBar1)
        self.SetMinSize(wx.Size(100, 100))
        self.SetAutoLayout(True)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(wx.EVT_IDLE, self.OnIdle)

        self.statusBar_main = wx.StatusBar(id=wxID_FRAMEMAINSTATUSBAR_MAIN,
              name=u'statusBar_main', parent=self,
              style=wx.THICK_FRAME | wx.ST_SIZEGRIP)
        self.statusBar_main.SetStatusText(u'Status...')
        self.statusBar_main.SetAutoLayout(True)
        self.SetStatusBar(self.statusBar_main)

        self.notebook = wx.Notebook(id=wxID_FRAMEMAINnotebook,
              name=u'notebook', parent=self, pos=wx.Point(0, 0),
              size=wx.Size(573, 629), style=0)
        self.notebook.SetMinSize(wx.Size(-1, -1))

        self._init_sizers()

    #--------------------------------------------------------------------
    def __init__(self, parent):
        self.last_ubmatrix_path = ''
        self.last_lattice_path = ''
        self.last_ldm_path = ''

        self._init_ctrls(parent)
        #Make the tabs for the notebook
        self.LoadNotebook()
        
        #Subscribe to messages
        model.messages.subscribe(self.OnStatusBarUpdate, model.messages.MSG_UPDATE_MAIN_STATUSBAR)
        model.messages.subscribe(self.OnScriptCommand, model.messages.MSG_SCRIPT_COMMAND)
        
        self.count = 0

        #Set the icon
        icon_file = os.path.join(os.path.dirname(__file__), CrystalPlan_version.icon_file)
        self.SetIcon( wx.Icon(icon_file, wx.BITMAP_TYPE_PNG) )
        
    #--------------------------------------------------------------------
    def LoadNotebook(self):
        """Add the notebook tabs. """
        self.tabs = []
        
        self.tab_startup = panel_startup.PanelStartup(parent=self.notebook)
        self.tab_sample = panel_sample.PanelSample(parent=self.notebook)
        self.tab_goniometer = panel_goniometer.PanelGoniometer(parent=self.notebook)
        self.tab_experiment = panel_experiment.PanelExperiment(parent=self.notebook)
        if not gui_utils.fourcircle_mode():
            self.tab_add = panel_add_positions.PanelAddPositions(parent=self.notebook)
            self.tab_try = panel_try_position.PanelTryPosition(parent=self.notebook)
            self.tab_detectors = panel_detectors.PanelDetectors(parent=self.notebook)


        def AddPage(tab, title, mac_title="", select=False):
            if (gui_utils.is_mac() or gui_utils.is_windows()) and not (mac_title==""):
                title = mac_title
            self.notebook.AddPage(tab, title, select)

        AddPage(self.tab_startup, 'Q-Space', 'Q-Space', select=True)
        if not gui_utils.fourcircle_mode():
            AddPage(self.tab_detectors, 'Detectors')
        AddPage(self.tab_goniometer, 'Goniometer')
        AddPage(self.tab_sample, 'Sample')
        if not gui_utils.fourcircle_mode():
            AddPage(self.tab_try, 'Try an\nOrientation', 'Try Orientation')
            AddPage(self.tab_add, 'Add\nOrientations', 'Add Orientations')
        AddPage(self.tab_experiment, 'Experiment\nPlan', 'Experiment Plan' )

        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGING, self.onNotebookPageChanging)


    #--------------------------------------------------------------------
    def RefreshAll(self):
        """Refresh all the tabs."""
        for i in xrange(self.notebook.GetPageCount()):
            tab = self.notebook.GetPage(i)
            #Call the refresh method, if it exists
            if hasattr(tab, "Refresh"):
                tab.Refresh()

    #--------------------------------------------------------------------
    #---------------- other event handlers ------------------------------
    #--------------------------------------------------------------------
    def onNotebookPageChanging(self, event):
        tab = self.notebook.GetCurrentPage()
        if not tab is None:
            if hasattr(tab, 'needs_apply'):
                if tab.needs_apply():
                    wx.MessageDialog(self, "You have changed some settings in this tab. \nYou need to click the 'Apply' button to apply them!", "Need To Apply Changes", wx.OK).ShowModal()

    
    def OnClose(self, event):
        res = wx.MessageDialog(self, "Are you sure you want to quit %s?" % CrystalPlan_version.package_name, "Quit %s?" % CrystalPlan_version.package_name, wx.YES_NO | wx.YES_DEFAULT).ShowModal()
        if res == wx.ID_YES:
            self.Destroy()
        else:
            event.Veto()

    def OnStatusBarUpdate(self, message):
        """Called when we receive a message that the statusbar needs updating."""
        #print message
        self.statusBar_main.SetStatusText(message.data)

    def OnScriptCommand(self, message):
        """Called to execute a scripted GUI command."""
        # @type call FunctionCall
        call = message.data
        #Call that function
        call.function(*call.args, **call.kwargs)

    def OnIdle(self, event):
        self.count += 1
        #print "Idle", self.count
        event.Skip()
        




#--------------------------------------------------------------------
if __name__ == '__main__':
    #Test routine
    model.instrument.inst = model.instrument.Instrument(model.config.cfg.default_detector_filename)
    model.experiment.exp = model.experiment.Experiment(model.instrument.inst)
    (app, frm) = gui_utils.test_my_gui(FrameMain)
    app.MainLoop()

