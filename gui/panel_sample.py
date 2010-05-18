#Boa:FramePanel:PanelSample
"""
Panel in the main window to set sample crystal characteristics.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import time
import numpy as np

#--- GUI Imports ---
import dialog_edit_crystal
import gui_utils
import display_thread

#--- Traits imports ---
from enthought.traits.api import HasTraits,Int,Float,Str,String,Property,Bool, List, Tuple, Array
from enthought.traits.ui.api import View,Item,Group,Label,Heading, Spring, Handler, TupleEditor, TabularEditor, ArrayEditor, TextEditor, CodeEditor
from enthought.traits.ui.menu import OKButton, CancelButton,RevertButton
from enthought.traits.ui.menu import Menu, Action, Separator

#--- Model Imports ---
import model
from model.crystals import Crystal

#=========================================================
#=========================================================
#=========================================================
[wxID_PANELSAMPLE, wxID_PANELSAMPLEBUTTONAPPLYRANGE, 
 wxID_PANELSAMPLEBUTTONEDITCRYSTAL, wxID_PANELSAMPLEBUTTONREVERTRANGE, 
 wxID_PANELSAMPLESTATICLINE1, wxID_PANELSAMPLESTATICTEXTRANGEHEADER, 
] = [wx.NewId() for _init_ctrls in range(6)]

#=========================================================
class HKLRangeSettings(HasTraits):
    """Simple class, with Traits, to set range in hkl."""
    h_range = Array( shape=(1,2), dtype=Int)
    k_range = Array( shape=(1,2), dtype=Int)
    l_range = Array( shape=(1,2), dtype=Int)
    automatic = Bool(False)
    limit_to_sphere = Bool(False)

    view = View( Item("h_range", enabled_when="not automatic"), Item("k_range", enabled_when="not automatic"), Item("l_range", enabled_when="not automatic"),
            Item("automatic", label="Automatically fit to experiment's min d-spacing?"),
            Item("limit_to_sphere", label="Limit to a sphere of radius corresponding to d_min?")
            )

    def __init__(self, exp, *args, **kwargs):
        """Constructor, read in the range values from the experiment exp."""
        HasTraits.__init__(self, *args, **kwargs)
        self.read_from_exp(exp)

    def read_from_exp(self, exp):
        """Read in the ranges from an Experiment instance. """
        self.h_range = np.array( exp.range_h ).reshape(1,2)
        self.k_range = np.array( exp.range_k ).reshape(1,2)
        self.l_range = np.array( exp.range_l ).reshape(1,2)
        self.automatic = exp.range_automatic
        self.limit_to_sphere = exp.range_limit_to_sphere

    def set_in_exp(self, exp):
        """Sets the ranges in an Experiment instance. """
        exp.range_h = tuple(self.h_range.flatten().astype(int))
        exp.range_k = tuple(self.k_range.flatten().astype(int))
        exp.range_l = tuple(self.l_range.flatten().astype(int))
        exp.range_automatic = self.automatic
        exp.range_limit_to_sphere = self.limit_to_sphere

    def is_valid(self):
        """Return True if the values entered make sense."""
        #The higher bound needs to be bigger for each.
        return self.automatic or \
                ((self.h_range[0,1] >=self.h_range[0,0]) and \
                (self.k_range[0,1] >= self.k_range[0,0]) and \
                (self.l_range[0,1] >= self.l_range[0,0]))

    def is_too_many(self):
        """Count the reflections in this range, warn if there are too many.

        Returns:
            bool: true if the number of reflections given by the selection is very large.
            n: integer, number of reflections that will be shown.
        """
        #TODO: Count automatic peaks
        if self.automatic:
            return (False, 1e3)
        
        n = (self.h_range[0,1]-self.h_range[0,0]+1) * \
            (self.k_range[0,1]-self.k_range[0,0]+1) * \
            (self.l_range[0,1]-self.l_range[0,0]+1)
        return ((n > 1e5), int(n))








#=========================================================
class PanelSample(wx.Panel):
    def _init_coll_boxSizerRangeButtons_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.buttonApplyRange, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)
        parent.AddWindow(self.buttonRevertRange, 0, border=0, flag=0)

    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.buttonEditCrystal, 0, border=0,
              flag=wx.ALIGN_CENTER)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticLine1, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextRangeHeader, 0, border=0, flag=0)
        parent.AddWindow(self.staticTextRangeHeader2, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddSizer(self.boxSizerRangeButtons, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=wx.EXPAND)
        parent.AddWindow(self.staticLine2, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)

        self.boxSizerRangeButtons = wx.BoxSizer(orient=wx.HORIZONTAL)

        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerRangeButtons_Items(self.boxSizerRangeButtons)

        self.SetSizer(self.boxSizerAll)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_PANELSAMPLE, name=u'PanelSample',
              parent=prnt, pos=wx.Point(647, 243), size=wx.Size(419, 467),
              style=wx.TAB_TRAVERSAL)
        self.SetClientSize(wx.Size(419, 467))

        self.buttonEditCrystal = wx.Button(id=wxID_PANELSAMPLEBUTTONEDITCRYSTAL,
              label=u'  Edit Crystal Parameters  ', name=u'buttonEditCrystal',
              parent=self, pos=wx.Point(117, 8), style=0)
        self.buttonEditCrystal.Bind(wx.EVT_BUTTON,
              self.OnButtonEditCrystalButton,
              id=wxID_PANELSAMPLEBUTTONEDITCRYSTAL)

        self.staticTextRangeHeader = wx.StaticText(id=wxID_PANELSAMPLESTATICTEXTRANGEHEADER,
              label=u'Enter the range of h, k, and l values to calculate:',
              name=u'staticTextRangeHeader', parent=self, pos=wx.Point(0, 45),
              size=wx.Size(371, 17), style=0)
        self.staticTextRangeHeader.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL,
              wx.BOLD, False, u'Sans'))

        self.staticTextRangeHeader2 = wx.StaticText(
              label=u'Values are inclusive',
              name=u'staticTextRangeHeader2', parent=self, pos=wx.Point(0, 45),
              size=wx.Size(371, 17), style=0)

        self.buttonApplyRange = wx.Button(id=wxID_PANELSAMPLEBUTTONAPPLYRANGE,
              label=u'  Apply New Range  ', name=u'buttonApplyRange', parent=self,
              pos=wx.Point(0, 78), style=0)
        self.buttonApplyRange.Bind(wx.EVT_BUTTON, self.OnButtonApplyRangeButton,
              id=wxID_PANELSAMPLEBUTTONAPPLYRANGE)

        self.buttonRevertRange = wx.Button(id=wxID_PANELSAMPLEBUTTONREVERTRANGE,
              label=u'  Revert  ', name=u'buttonRevertRange', parent=self,
              pos=wx.Point(193, 78), style=0)
        self.buttonRevertRange.Bind(wx.EVT_BUTTON,
              self.OnButtonRevertRangeButton,
              id=wxID_PANELSAMPLEBUTTONREVERTRANGE)

        self.staticLine1 = wx.StaticLine(id=wxID_PANELSAMPLESTATICLINE1,
              name='staticLine1', parent=self, pos=wx.Point(0, 115),
              size=wx.Size(419, 2), style=0)

        self.staticLine2 = wx.StaticLine(name='staticLine2', parent=self, pos=wx.Point(0, 115), size=wx.Size(419, 2), style=0)

        self._init_sizers()


    def __init__(self, parent):
        self._init_ctrls(parent)

        #Make a simple, mostly read-only view for the crystal
        crystal_view = View(
            Item("name", label="Crystal Name"),
            Item("description", label="Description:", editor=TextEditor(multi_line=True)),
            Item("lattice_lengths_arr", label="Lattice sizes (Angstroms)", format_str="%.3f", style='readonly'),
            Item("lattice_angles_deg_arr", label="Lattice angles (degrees)", format_str="%.3f", style='readonly'),
            Item("ub_matrix", label="Sample's UB Matrix", style='readonly', format_str="%9.5f"),
            Item("point_group_name", label="Point Group", style='readonly'),
            Item("recip_a", label="a*", style='readonly'),
            Item("recip_b", label="b*", style='readonly'),
            Item("recip_c", label="c*", style='readonly'),
            resizable=True
            )
        #Make it into a control
        crystal = model.experiment.exp.crystal
        self.crystal_control = crystal.edit_traits(parent=self, view=crystal_view, kind='subpanel').control
        self.boxSizerAll.Insert(0, self.crystal_control, 0, border=1, flag=wx.EXPAND)
        self.GetSizer().Layout()
        #Create the range settings object using the global experiment
        self.range_settings = HKLRangeSettings(model.experiment.exp)
        self.range_control = self.range_settings.edit_traits(parent=self, kind='subpanel').control
        self.boxSizerAll.Insert(9, self.range_control, 0, border=1, flag=wx.EXPAND)

    #---------------------------------------------------------------------------
    def apply_crystal_range(self):
        #Apply the range to the experiment
        self.range_settings.set_in_exp(model.experiment.exp)

        #Make a progress bar
        self.count = 4
        max = len(model.instrument.inst.positions)+2 #Steps in calculation
        self.prog_dlg = wx.ProgressDialog( "Reflection Calculation Progress",        "Initializing reflections for sample.              ",
            max, style = wx.PD_CAN_ABORT | wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME |
                         wx.PD_ESTIMATED_TIME | wx.PD_REMAINING_TIME | wx.PD_AUTO_HIDE)
        self.prog_dlg.Update(self.count)

        #Initialize the peaks here.
        model.experiment.exp.initialize_reflections()

        #In automatic mode, lets read out what the hkl range was
        if self.range_settings.automatic:
            self.range_settings.read_from_exp(model.experiment.exp)

        #Recalculate peaks here.
        model.experiment.exp.recalculate_reflections(None, calculation_callback=self._calculation_progress_report)
        
        #Clean up progress dialog
        self.prog_dlg.Destroy()

        #Manually send message to redraw
        model.messages.send_message(model.messages.MSG_EXPERIMENT_REFLECTIONS_CHANGED)
        
    #---------------------------------------------------------------------------
    def _calculation_progress_report(self, poscov):
        """Callback to show progress during a calculation."""
        self.count += 1
        self.prog_dlg.Update(self.count, "Calculating reflections for orientation %s..." % (model.instrument.inst.make_angles_string(poscov.angles)))


    #---------------------------------------------------------------------------
    def OnButtonEditCrystalButton(self, event):
        """Clicking the button to change the crystal settings."""
        old_U = model.experiment.exp.crystal.get_u_matrix()
        
        if dialog_edit_crystal.show_dialog(self, model.experiment.exp.crystal):
            #User clicked okay, something (proably) changed
            new_U = model.experiment.exp.crystal.get_u_matrix()
            if not np.allclose(old_U, new_U):
                #The sample mounting U changed, so we need to recalc all the 3D volume coverage
                gui_utils.do_recalculation_with_progress_bar(new_U)
                #Send message signaling a redraw of volume plots
                display_thread.handle_change_of_qspace(changed_sample_U_matrix=new_U)

            #Update the hkl range settings, especially for automatic settings
            self.apply_crystal_range()
        event.Skip()


    #---------------------------------------------------------------------------
    def OnButtonApplyRangeButton(self, event):
        #Are they okay?
        if not self.range_settings.is_valid():
            wx.MessageDialog(self, "Invalid entries in the ranges. Make sure the higher bound is >= the lower bound.", "Can't apply ranges", wx.OK | wx.ICON_ERROR).ShowModal()
            return
        (b,n) = self.range_settings.is_too_many()
        if b:
            dlg =  wx.MessageDialog(self, "The number of reflections given by this range, %d, is very large. Are you sure?" % n,
                                    "Too Many Reflections", wx.YES_NO | wx.ICON_INFORMATION)
            res = dlg.ShowModal()
            dlg.Destroy()
            if res != wx.ID_YES:
                return
        #Do it!
        self.apply_crystal_range()
        event.Skip()


    def OnButtonRevertRangeButton(self, event):
        #Go back to what is saved in there.
        self.range_settings.read_from_exp(model.experiment.exp)
        event.Skip()







if __name__=="__main__":
    model.crystals.initialize()
    #Test routine
    model.instrument.inst = model.instrument.Instrument()
    model.experiment.exp = model.experiment.Experiment(model.instrument.inst)
    import gui_utils
    (app, pnl) = gui_utils.test_my_gui(PanelSample)
    app.MainLoop()

    