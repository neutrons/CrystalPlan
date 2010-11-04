#Boa:FramePanel:PanelDetectors
"""
PanelDetectors: a GUI component that allows the user to select which detectors are used in the
coverage calculations.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import os
import numpy as np

#--- GUI Imports ---
import display_thread
import gui_utils
from enthought.traits.api import HasTraits, Range, Instance, on_trait_change
from enthought.traits.ui.api import View, Item, HGroup
try:
    from mlab_utils import *
    from enthought.tvtk.pyface.scene_editor import SceneEditor
    from enthought.mayavi.tools.mlab_scene_model import MlabSceneModel
    from enthought.mayavi.core.ui.mayavi_scene import MayaviScene
    from enthought.mayavi.core.scene import Scene
    from enthought.mayavi.api import Engine
    from enthought.mayavi.tools.engine_manager import engine_manager
except ImportError, e:
    print "PanelDetectors: ERROR IMPORTING MAYAVI - 3D WILL NOT WORK!"

#--- Model Imports ---
import model



#================================================================================================
#================================================================================================
#================================================================================================
class DetectorVisualization(HasTraits):
    """3-D, real-space view of detector geometry."""
#    # Scene being displayed.
#    scene = Instance(MlabSceneModel, ())
#
#    # The layout of the panel created by Traits
#    view = View(Item('scene', editor=SceneEditor(), resizable=True,
#                    show_label=False), resizable=True)

    #Make an engine just for this frame
    #engine = engine_manager.new_engine() # Instance(Engine, args=())
    engine = Instance(Engine, args=())

    #Create the scene
    scene = Instance(MlabSceneModel, ())

    def _scene_default(self):
        """Default initializer for scene"""
        self.engine.start()
        scene = MlabSceneModel(engine=self.engine)
        #self.engine.scenes.append(scene) #Make sure the engine manager nows about it
        return scene

    #3D view of the scene.
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), resizable=True,
                    show_label=False), resizable=True)
                    
    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)

    #---------------------------------------------------------------------------------------------
    def detector_plot(self, det, color=(1,1,1)):
        """Plot the location of the detector in real space, as an outline."""
        col=color
        rad=None
        #Plot between each corner and connect back to 1st corner
        lines(det.corners+[det.corners[0]], color=col, tube_radius=rad, mlab=self.scene.mlab)
        #Plot the normal, out of the center
#        lines([det.base_point.flatten(), (det.base_point+det.normal*det.width/2).flatten()], color=col, tube_radius=rad, mlab=self.scene.mlab)
        #Find the middle and put text there
        center = det.pixels[:, det.xpixels/2,  det.ypixels/2]
        text3d(center, det.name, font_size=15, color=col, mlab=self.scene.mlab)

    #---------------------------------------------------------------------------------------------
    def display(self, reset_view=True):
        """Plot all the detectors in real space."""
        #Map mlab to the scene in the control.
        mlab = self.scene.mlab
        self.scene.disable_render = True #Render without displaying, to speed it up.
        mlab.clf()
        mlab.plot3d([0,1], [0,1], [0,1]) 
    
        mlab.options.offscreen = False
        #Make a simple color map
        c = 0
        max_distance = 100
        for det in model.instrument.inst.detectors:
            c = c+1
            n=3.0
            r = (c%n)/n
            g = ((c/n)%n)/n
            b = ((c/(n*n))%n)/n
            col = (r, g, b)
            self.detector_plot(det, col)
            #Find the largest distance between the detector
            max_distance = max(max_distance, det.distance)

        #Add arrow indicating neutron beam direction
        length = max_distance * 1.25
        col = (1.0,0.0,0.0)
        arrow( vector([0,0,-length]),  vector([0,0,+length]), head_size=length/10, color=col, tube_radius=length/100., mlab=mlab)
        text3d(vector([0,0,+length*1.1]), "Beam direction", font_size=18, color=(0,0,0), mlab=mlab)
        arrow( vector([0,0,0]),  vector([0,+length,0]), head_size=length/12, color=(0,0,0), tube_radius=length/150., mlab=mlab)
        text3d(vector([0,+length*1.1,0]), "Up", font_size=16, color=(0,0,0), mlab=mlab)
        mlab.orientation_axes()
        if reset_view:
            #Set the camera
            mlab.view(56.372692217264493, 43.295412845543929, 2457.6030853527195, focalpoint=np.zeros(3))
            mlab.roll(2.6)

        self.scene.disable_render = False

    #-----------------------------------------------------------------------------------------------
    def cleanup(self):
        """Clean-up on closing the form"""
        #Remove the scene from engine
        self.engine.remove_scene(self.scene)
        self.engine.stop()


#================================================================================================
#================================================================================================
#================================================================================================
class DetectorView3D(wx.Frame):
    """Frame that will hold a 3D view of all the detectors."""
                    
    #---------------------------------------------------------------------------------------------
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, 'Direct Space 3D View of Detectors', size=wx.Size(600, 500),
        pos=wx.Point(100,50))
        self.visualization = DetectorVisualization()
        #Create a control in the frame
        self.control = self.visualization.edit_traits(parent=self,
                                kind='subpanel').control
        self.Show()
        self.update(reset_view=True)
        #Close event
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    #---------------------------------------------------------------------------------------------
    def update(self, reset_view=False):
        """Update the frame (and 3D view)"""
        self.visualization.display(reset_view)

    #---------------------------------------------------------------------------------------------
    def OnClose(self, event):
        #print "Detectors 3D view was", self.visualization.scene.mlab.view(), "with a roll of", self.visualization.scene.mlab.roll()
        #Clear the attribute from the parent
        self.GetParent().frame3d = None
        self.visualization.cleanup()
        event.Skip()



#================================================================================================
#================================================================================================
#================================================================================================
class DetectorListController:
    """Displays the detectors as a checked list."""
    lst = None

    #An array of the % covered by each individual detector.
    detector_coverage = None

    #----------------------------------------------------------------------------------------
    def __init__(self, panel):
        #UI list box
        self.panel = panel
        self.lst = panel.chklistDetectors
        #Subscribe to changes to detector list.
        model.messages.subscribe(self.update, model.messages.MSG_DETECTOR_LIST_CHANGED)
        self.update()

    #----------------------------------------------------------------------------------------
    def load_detectors_dialog(self):
        """Open a dialog to load detectors from a CSV/detcal file."""
        filename = model.config.cfg.default_detector_filename
        (path, ignored) = os.path.split( os.path.abspath(filename) )
        filters = 'CSV or detcal files|*.csv;*.detcal|CSV files (*.csv)|*.csv|detcal files (*.detcal)|*.csv|All files (*)|*|'
        if gui_utils.is_mac():
            filters = '' #This is needed on Mac, for some reason the filters crashes otherwise.
        print 'opening dialog for path', path, filename
        dialog = wx.FileDialog ( None, defaultFile=filename, defaultDir=path, message='Choose a .csv or .detcal file describing the detector geometry', wildcard=filters, style=wx.OPEN )
        dialog.SetFilterIndex(0)
        if dialog.ShowModal() == wx.ID_OK:
            filename = dialog.GetPath()
            dialog.Destroy()
        else:
            #'Nothing was selected.
            dialog.Destroy()
            return
        #Do load the file
        self.load_detector_file(filename)

    #----------------------------------------------------------------------------------------
    def load_detector_file(self, filename):
        """Load detectors from a file."""
        #Load if it exists
        if not os.path.exists(filename):
            wx.MessageDialog(self, "File '%s' was not found!" % filename, 'Error', wx.OK | wx.ICON_ERROR).ShowModal()
        else:
            old_detectors = model.instrument.inst.detectors
            try:
                model.instrument.inst.load_detectors_file(filename)
                #Save as the next default
                model.config.cfg.default_detector_filename = filename
                #Send a message saying that the detector list has changed
                model.messages.send_message(model.messages.MSG_DETECTOR_LIST_CHANGED)

                #Recalculate everything!
                gui_utils.do_recalculation_with_progress_bar(new_sample_U_matrix=None)
                #Update displays
                display_thread.handle_change_of_qspace()
                if not self.panel.frame3d is None:
                    pass
                    #The 3D view is open, update that by closing and re-opening
                    wx.CallAfter(self.panel.frame3d.Close)
#                    #Open a new one!
#                    wx.CallAfter(self.panel.OnButton_view_detectorsButton, None)
#                    #self.panel.frame3d.update(reset_view=False)

            except:
                #Go back to the old list
                model.instrument.inst.detectors = old_detectors
                #Re-raise the error
                raise


    #----------------------------------------------------------------------------------------
    def update(self, *args):
        """Display the detector list"""
        #Make the list
        items = list()
        
        #Do we display the detector coverage?
        use_coverage = False
        if not (self.detector_coverage is None):
            use_coverage = (len(self.detector_coverage) == len(model.instrument.inst.detectors))

        #Add each item to the list
        for i in range( len(model.instrument.inst.detectors) ):
            det = model.instrument.inst.detectors[i]
            s = "%7s" % (det.name)
            if use_coverage: s = s + (" : %6.2f%%" % self.detector_coverage[i])
            items.append(s)
            
        #Sets the list in a single shot.
        self.lst.Set(items)
        
        #Check the boxes as appropriate
        for x in range(len(model.instrument.inst.detectors)):
            #Look up which have been selected in the latest parameters
            self.lst.Check(x, display_thread.is_detector_selected(x))

    #----------------------------------------------------------------------------------------
    def select_all(self, value):
        """Select or deselect all items in the checkboxlist."""
        for x in range(self.lst.GetCount()):
            self.lst.Check(x, value)
        #Update as needed
        self.changed()

    #----------------------------------------------------------------------------------------
    def changed(self):
        """Called when there is a change in the checked list box selection."""
        #Make the list of true/false bools
        detlist = list()
        for x in range(self.lst.GetCount()):
            detlist.append(self.lst.IsChecked(x))
        
        det = model.experiment.ParamDetectors(detlist)
        #Save that object as the parameter to change
        display_thread.NextParams[model.experiment.PARAM_DETECTORS] = det

    #----------------------------------------------------------------------------------------
    def calculate_stats(self):
        """Calculate the coverage of each individual detector."""
        self.detector_coverage = np.zeros( self.lst.GetCount() )
        for x in range(self.lst.GetCount()):
            #Make an empty bool array
            detectors_used =  np.zeros( self.lst.GetCount() )
            #Except for a single detector.
            detectors_used[x] = 1
            #Calculate for this detector and all positions
            qspace = model.experiment.exp.inst.total_coverage(detectors_used, None)
            #And calculate the % coverage.
            (coverage_percent, redundant_percent) = model.experiment.exp.overall_coverage_stats(qspace)
            self.detector_coverage[x] = coverage_percent
            print "Detector #%s has %s coverage." % (x, coverage_percent)
        #And redo the list
        self.update()


    #----------------------------------------------------------------------------------------
    def select_detector_list(self, det_list):
        """Select a list of detectors.
        det_list: 1-based list of detector numbers."""
        
        #Check or uncheck as needed
        count = 0
        for x in range(self.lst.GetCount()):
            self.lst.Check(x, ((x+1) in det_list) )

        #Redo the list
        self.changed()
        
    #----------------------------------------------------------------------------------------
    def select_best_detectors(self, num):
        """Sort the coverage of each detector and pick the best 'num' values."""

        #Recalculate the statistics if needed
        recalc = False
        if self.detector_coverage is None:
            recalc = True
        else:
            recalc = len(self.detector_coverage) != len(model.experiment.exp.inst.detectors)
        if recalc:
            self.calculate_stats()

        #Sort the coverage
        temp = sorted(self.detector_coverage, reverse=True)

        #Adjust inputs
        if num < 1: num = 1
        if num > len(temp): num = len(temp)
        #What's the worst coverage value we keep?
        worst_coverage = temp[num-1]

        #Select the items that are as good or better
        print "Selecting the best", num, "detectors:",
        count = 0
        for x in range(self.lst.GetCount()):
            b = (self.detector_coverage[x] >= worst_coverage)
            if b: count = count + 1
            if count > num: b = False
            if b:
                print model.experiment.exp.inst.detectors[x].name,
            self.lst.Check(x,  b )
        print ""

        #Redo the list
        self.changed()






#================================================================================================
[wxID_PANELDETECTORS, wxID_PANELDETECTORSBUTTONOPTIMIZE, 
 wxID_PANELDETECTORSBUTTONSELECTBEST, wxID_PANELDETECTORSBUTTONSELECTLIST, 
 wxID_PANELDETECTORSBUTTONSTATS, wxID_PANELDETECTORSBUTTON_VIEW_DETECTORS, 
 wxID_PANELDETECTORSCHECKSELECTALL, wxID_PANELDETECTORSCHKLISTDETECTORS, 
 wxID_PANELDETECTORSSTATICTEXT1, 
] = [wx.NewId() for _init_ctrls in range(9)]

class PanelDetectors(wx.Panel):
    def _init_coll_boxSizer3_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.buttonStats, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.buttonSelectBest, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.button_view_detectors, 0, border=0, flag=0)

    def _init_coll_boxSizer_main_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(4, 4), border=0, flag=0)
        parent.AddWindow(self.staticText1, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.buttonLoadDetectors, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.boxSizer3, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddSizer(self.boxSizer2, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.chklistDetectors, 1, border=0, flag=wx.EXPAND)

    def _init_coll_boxSizer2_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.checkSelectAll, 0, border=0, flag=wx.ALIGN_CENTER_VERTICAL)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.buttonSelectList, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.buttonOptimize, 0, border=0, flag=0)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizer_main = wx.BoxSizer(orient=wx.VERTICAL)

        self.boxSizer2 = wx.BoxSizer(orient=wx.HORIZONTAL)

        self.boxSizer3 = wx.BoxSizer(orient=wx.HORIZONTAL)

        self._init_coll_boxSizer_main_Items(self.boxSizer_main)
        self._init_coll_boxSizer2_Items(self.boxSizer2)
        self._init_coll_boxSizer3_Items(self.boxSizer3)

        self.SetSizer(self.boxSizer_main)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_PANELDETECTORS, name=u'PanelDetectors',
              parent=prnt, pos=wx.Point(755, 544), size=wx.Size(512, 511),
              style=wx.TAB_TRAVERSAL)
        self.SetClientSize(wx.Size(512, 511))
        self.SetAutoLayout(True)

        self.staticText1 = wx.StaticText(id=wxID_PANELDETECTORSSTATICTEXT1,
              label=u'Check which detectors to consider in the calculation.',
              name='staticText1', parent=self, pos=wx.Point(0, 0), style=0)
        self.staticText1.SetToolTipString(u'Detectors!')
        self.staticText1.SetMinSize(wx.Size(-1, -1))

        self.button_view_detectors = wx.Button(id=wxID_PANELDETECTORSBUTTON_VIEW_DETECTORS,
              label=u'  View Detectors in 3D  ', name=u'button_view_detectors',
              parent=self, pos=wx.Point(321, 62), style=0)
        self.button_view_detectors.Bind(wx.EVT_BUTTON,
              self.OnButton_view_detectorsButton,
              id=wxID_PANELDETECTORSBUTTON_VIEW_DETECTORS)

        self.chklistDetectors = wx.CheckListBox(choices=[],
              id=wxID_PANELDETECTORSCHKLISTDETECTORS, name=u'chklistDetectors',
              parent=self, pos=wx.Point(0, 99), size=wx.Size(512, 412),
              style=wx.NO_3D)
        self.chklistDetectors.SetMinSize(wx.Size(20, 40))
        self.chklistDetectors.SetFont(wx.Font(10, 76, wx.NORMAL, wx.NORMAL,
              False, u'Monospace'))
        self.chklistDetectors.Bind(wx.EVT_CHECKLISTBOX,
              self.OnChklistDetectorsChecklistbox,
              id=wxID_PANELDETECTORSCHKLISTDETECTORS)

        self.checkSelectAll = wx.CheckBox(id=wxID_PANELDETECTORSCHECKSELECTALL,
              label=u'Select All', name=u'checkSelectAll', parent=self,
              pos=wx.Point(0, 25), style=0)
        self.checkSelectAll.SetValue(False)
        self.checkSelectAll.SetMinSize(wx.Size(-1, 24))
        self.checkSelectAll.Bind(wx.EVT_CHECKBOX, self.OnCheckSelectAllCheckbox,
              id=wxID_PANELDETECTORSCHECKSELECTALL)

        self.buttonStats = wx.Button(id=wxID_PANELDETECTORSBUTTONSTATS,
              label=u'  Coverage Stats...  ', name=u'buttonStats', parent=self,
              pos=wx.Point(0, 62),  style=0)
        self.buttonStats.Bind(wx.EVT_BUTTON, self.OnButtonStatsButton,
              id=wxID_PANELDETECTORSBUTTONSTATS)

        self.buttonSelectBest = wx.Button(id=wxID_PANELDETECTORSBUTTONSELECTBEST,
              label=u'  Select best # detectors...  ', name=u'buttonSelectBest',
              parent=self, pos=wx.Point(128, 62), style=0)
        self.buttonSelectBest.Bind(wx.EVT_BUTTON, self.OnButtonSelectBestButton,
              id=wxID_PANELDETECTORSBUTTONSELECTBEST)

        self.buttonSelectList = wx.Button(id=wxID_PANELDETECTORSBUTTONSELECTLIST,
              label=u'  Select List...  ', name=u'buttonSelectList', parent=self,
              pos=wx.Point(93, 25), style=0)
        self.buttonSelectList.Bind(wx.EVT_BUTTON, self.OnButtonSelectListButton,
              id=wxID_PANELDETECTORSBUTTONSELECTLIST)

        self.buttonOptimize = wx.Button(id=wxID_PANELDETECTORSBUTTONOPTIMIZE,
              label=u'  Optimize using GA  ', name=u'buttonOptimize', parent=self,
              pos=wx.Point(207, 25), style=0)
        self.buttonOptimize.Bind(wx.EVT_BUTTON, self.OnButtonOptimizeButton,
              id=wxID_PANELDETECTORSBUTTONOPTIMIZE)

        self.buttonLoadDetectors = wx.Button(label=u'  Load Detectors CSV/detcal file  ',
                name=u'buttonLoadDetectors', parent=self,
                pos=wx.Point(207, 25), style=0)
        self.buttonLoadDetectors.Bind(wx.EVT_BUTTON, self.OnButtonLoadDetectors)

        self._init_sizers()

    def __init__(self, parent):
        #No 3D frame to start
        self.frame3d = None

        self._init_ctrls(parent)
        #--- Additional code ---
        #Set up the View objects
        self.controller = DetectorListController(self)

    def Refresh(self):
        self.controller.update()

    def OnButton_view_detectorsButton(self, event):
        if self.frame3d is None:
            #Create the 3D frame
            self.frame3d = DetectorView3D(self, wx.NewId())
        else:
            #Raise to top
            self.frame3d.Raise()
            #Just update
            #self.frame3d.update(reset_view=False)
        if not event is None: event.Skip()
        
    def OnChklistDetectorsChecklistbox(self, event):
        self.controller.changed()
        event.Skip()

    def OnCheckSelectAllCheckbox(self, event):
        self.controller.select_all(self.checkSelectAll.GetValue())
        event.Skip()

    def OnButtonStatsButton(self, event):
        self.controller.calculate_stats()
        event.Skip()

    def OnButtonSelectBestButton(self, event):
        #Now prompt the user
        dlg = wx.TextEntryDialog(None, 'This will select the detectors with the highest coverage (by themselves).\nHow many detectors do you want to select?', 'Select Best Detectors', '')
        if dlg.ShowModal() == wx.ID_OK:
            #User did not cancel
            s = dlg.GetValue()

            try:
                val = int(s)
                #Now do the selection.
                self.controller.select_best_detectors(val)
            except:
                wx.MessageDialog(None, "Error! Could not convert '%s' to integer." % s, 'Error', wx.OK | wx.ICON_ERROR).ShowModal()

        #If cancelled or not, you need to destroythe dialog.
        dlg.Destroy()


    def OnButtonSelectListButton(self, event):
        """Select a list of detectors."""
        dlg = wx.TextEntryDialog(None, 'Type in or paste the list of detector numbers, e.g. "[1, 2, 4]" :', 'Select List of Detectors', '[1]')
        if dlg.ShowModal() == wx.ID_OK:
            s = dlg.GetValue()
            dets = eval("list(%s)" % s)
            self.controller.select_detector_list(dets)
        dlg.Destroy()
        event.Skip()

    def OnButtonOptimizeButton(self, event):
        """Start optimizing using GA."""
        dop = model.optimize_coverage.DetectorOptimizationParameters()
        if dop.configure_traits():
            #User clicked OKAY, do optimization
            det_list = model.optimize_coverage.optimize_detector_choice(dop, gui=True)
            #This will check the boxes as well as update the other ui windows.
            self.controller.select_detector_list(det_list)
        event.Skip()

    def OnButtonLoadDetectors(self, event):
        self.controller.load_detectors_dialog()
        event.Skip()


#====================================================================================
if __name__ == "__main__":

    model.instrument.inst = model.instrument.Instrument()
    model.goniometer.initialize_goniometers()
#    frame3d = DetectorView3D(self, wx.NewId())
#    frame3d.display()
#    sys.exit()

    import gui_utils
    (app, pnl) = gui_utils.test_my_gui(PanelDetectors)
    app.MainLoop()
