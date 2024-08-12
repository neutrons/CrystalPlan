"""
GUI to set crystal parameters, as a dialog.
"""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import os
import copy
import numpy as np
import sys

#--- Traits imports ---
from traits.api import HasTraits,Int,Float,Str,String,Property,Bool, List, Tuple, Array
from traitsui.api import View,Item,Group,Label,Heading, Spring, Handler, TupleEditor, TabularEditor, ArrayEditor, TextEditor, CodeEditor
from traitsui.editors import EnumEditor
from traitsui.menu import OKButton, CancelButton

#--- GUI Imports ---

#--- Model Imports ---
if __name__=="__main__":    sys.path.insert(0, "..")
import model
from model.crystals import Crystal

# ===========================================================================================
class SampleOrientationWhenUBMatrixWasSaved(HasTraits):
    """Small Traits class to prompt the user to enter the goniometer settings used
    when data was acquired (for the purposes of loading an ISAW UB matrix."""

    phi_degrees = Float(0.)
    chi_degrees = Float(0.)
    omega_degrees = Float(0.)
    
    view = View(
        Item('phi_degrees', label='Phi (degrees)'),
        Item('chi_degrees', label='Chi (degrees)'),
        Item('omega_degrees', label='Omega (degrees)'),
        buttons=[OKButton, CancelButton]
        )

# ===========================================================================================
class CrystalEditorTraitsHandler(Handler):
    """Handler that reacts to changes in the Crystal object."""

    def __init__(self, frame, *args, **kwargs):
        Handler.__init__(self, *args, **kwargs)
        self.add_trait('frame', frame)
        self.add_trait('user_clicked_okay', False)

    #---------------------------------------------------------------------------
    def setattr(self, info, object, name, value):
        """Called when any attribute is set."""
        #The parent class actually sets the value
        Handler.setattr(self, info, object, name, value)
        #Do other checks
        self.check_validity()

    #---------------------------------------------------------------------------
    def check_validity(self):
        #Show a warning for bad angles
        if self.frame.crystal.is_lattice_valid():
            self.frame.crystal.valid_parameters_yesno = "Yes"
        else:
            self.frame.crystal.valid_parameters_yesno = "No! Check the angles/lengths."
        #Make sure the reciprocal is calculated
        self.frame.crystal.calculate_reciprocal()
        #Make sure layout is okay. Might not be needed
        self.frame.GetSizer().Layout()

    #---------------------------------------------------------------------------
    def close(self, apply_change):
        """Close """
        if apply_change:
            #Modify the crystal data in the experiment.
            self.frame.original_crystal.copy_traits(self.frame.crystal)
        #Close it
        self.user_clicked_okay = apply_change
        self.frame.Close()



# ===========================================================================================
class DialogEditCrystal(wx.Dialog):
    """Dialog to edit a crystal's parameters."""

    def _init_coll_boxSizerButtons_Items(self, parent):
        parent.AddStretchSpacer(1)
        parent.AddWindow(self.buttonOK, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)
        parent.AddWindow(self.buttonCancel, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)

    def _init_coll_boxSizerAll_Items(self, parent):
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)
        parent.AddSizer(self.boxSizerButtons, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)

    def _init_coll_boxSizerIsaw_Items(self, parent):
        parent.AddSpacer(wx.Size(4, 4), border=0, flag=0)
        parent.AddWindow(self.staticTextHelp1,0, border=0, flag=wx.CENTER)
        parent.AddSpacer(wx.Size(4, 4), border=0, flag=0)
        parent.AddWindow(self.staticTextHelp3,0, border=0, flag=wx.LEFT)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextHelp2,0, border=0, flag=wx.CENTER)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)
        parent.AddWindow(self.control_load_angles,0, border=0, flag=wx.CENTER)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)
        parent.AddWindow(self.buttonReadUB, 0, border=0, flag=wx.CENTER)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)
        self.panelIsaw.SetSizer(self.boxSizerIsaw)

    def _init_sizers(self):
        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerIsaw_Items(self.boxSizerIsaw)
        self._init_coll_boxSizerButtons_Items(self.boxSizerButtons)
        self.SetSizer(self.boxSizerAll)


    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Dialog.__init__(self, name=u'DialogEditCrystal',
              parent=prnt, pos=wx.Point(702, 235), size=wx.Size(475, 600),
              style= wx.RESIZE_BORDER | wx.DEFAULT_DIALOG_STYLE | wx.MAXIMIZE_BOX | wx.MINIMIZE_BOX,
              title=u'Edit Crystal Parameters')
        self.SetClientSize(wx.Size(500, 800))

        self.buttonOK = wx.Button(
              label=u'Ok', name=u'buttonOK', parent=self,
              pos=wx.Point(16, 563), size=wx.Size(150, 29), style=0)
        self.buttonOK.Bind(wx.EVT_BUTTON, self.OnbuttonOKButton)

        self.buttonCancel = wx.Button(
              label=u'Cancel', name=u'buttonCancel', parent=self, pos=wx.Point(309,
              563), size=wx.Size(150, 29), style=0)
        self.buttonCancel.Bind(wx.EVT_BUTTON, self.OnbuttonCancelButton)

        self.notebook = wx.Notebook(name=u'notebook', parent=self, pos=wx.Point(100, 100), size=wx.Size(200,200), style=wx.TAB_TRAVERSAL)
        self.notebook.SetMinSize(wx.Size(-1, -1))
        self.notebook.Show()

        #--- ISAW load panel ----
        self.panelIsaw = wx.Panel(id=wx.NewId(), name="panelIsaw", parent=self.notebook, style=wx.TAB_TRAVERSAL)

        self.staticTextHelp1 = wx.StaticText(label=u'Loading an ISAW UB matrix file:',
            parent=self.panelIsaw, pos=wx.Point(166, 563), style=wx.ALIGN_CENTER)
        self.staticTextHelp1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, False, u'Sans'))
        self.staticTextHelp2 = wx.StaticText(label=u'ISAWev only!: before loading the file, enter the goniometer settings at the time you acquired the data from which this UB matrix was taken.\n\nFor ISAW, leave all angles at 0.0 (the UB matrix is already corrected by ISAW for the goniometer settings)',
            name=u'staticTextHelp2', parent=self.panelIsaw, pos=wx.Point(166, 563), style=wx.ALIGN_LEFT)
        self.staticTextHelp3 = wx.StaticText(label=u"Use ISAW's Initial Peaks Wizard or ISAWev's Find Peaks/Index Peaks functions to find the UB matrix and save it to a text file.",
            name=u'staticTextHelp2', parent=self.panelIsaw, pos=wx.Point(166, 563), style=wx.ALIGN_CENTER)
        #Wrap the text
        w = self.GetSize()[0]-60
        self.staticTextHelp2.Wrap(w)
        self.staticTextHelp3.Wrap(w)

        self.buttonReadUB = wx.Button(label=u'Read UB matrix from file...', name=u'buttonReadUB', parent=self.panelIsaw,
              pos=wx.Point(16, 563), size=wx.Size(250, 29), style=0)
        self.buttonReadUB.Bind(wx.EVT_BUTTON, self.OnButtonReadUB)

        #--- Manual panel ----
        self.panelManual = wx.Panel(id=wx.NewId(), name="panelManual", parent=self.notebook, style=wx.TAB_TRAVERSAL)
        self.buttonGenerateUB = wx.Button(label=u'Generate UB matrix', name=u'buttonGenerateUB', parent=self.panelManual,
              pos=wx.Point(16, 563), size=wx.Size(250, 29), style=0)
        self.buttonGenerateUB.Bind(wx.EVT_BUTTON, self.OnButtonGenerateUB)

        self.notebook.AddPage(self.panelIsaw, "Load from ISAW", select=True)


    #---------------------------------------------------------------------------
    def __init__(self, parent, my_crystal):
        self._init_ctrls(parent)
        
        #Make all the sizers
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerIsaw = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerManual = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerButtons = wx.BoxSizer(orient=wx.HORIZONTAL)

        #Setup the parameter editor traits ui panel
        self.original_crystal = my_crystal
        self.crystal = copy.copy(my_crystal)
        self.handler = CrystalEditorTraitsHandler(self)


        #Make the TRAITS view

        #At the top is a general view
        self.view_top =  View( Group(
            Item("name", label="Crystal Name"),
            Item("description", label="Description:", editor=TextEditor(multi_line=True)),
            Item("lattice_lengths_arr", label="Lattice sizes (Angstroms)", format_str="%.3f", style='readonly'),
            Item("lattice_angles_deg_arr", label="Lattice angles (degrees)", format_str="%.3f", style='readonly'),
            Item("point_group_name", label="Point Group:", editor=EnumEditor(name="point_group_name_list")),
            Item("reflection_condition_name", label="Reflection Condition:", editor=EnumEditor(name="reflection_condition_name_list")),
            Item("ub_matrix", label="Sample's UB Matrix", style='readonly', format_str="%9.5f"),
            Item("ub_matrix_is_from", label="UB matrix obtained from", style='readonly'),
            label="" ),
            )
        #Make the control and add it
        self.control_top = self.crystal.edit_traits(parent=self, view=self.view_top, kind='subpanel', handler=self.handler).control
        self.boxSizerAll.AddWindow(self.control_top, 0, border=0, flag=wx.EXPAND)

        # Now add the notebook, make it expand
        self.boxSizerAll.AddWindow(self.notebook, 1, border=4, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP)
            
        #Now a view to manually create the parameters
        angle_label = Group(
                    Label(label="""If all the sample mounting angles are zero, the sample's crystal
lattice coordinate system is aligned with the instrument coordinates.
The 'a' vector is parallel to x; 'b' is in the XY plane towards +y;
'c' goes towards +z.
""",
                    emphasized=False, show_label=True),
                    )

        fmt = "%.3f"
        self.view_manual = View(
        Group(
            Item("lattice_lengths_arr", label="Lattice dimensions\n(a,b,c - in Angstroms)", format_str=fmt),
            Item("lattice_angles_deg_arr", label="Lattice angles\n(alpha,beta,gamma - in degrees)", format_str=fmt),
            Item("valid_parameters_yesno", label="Lattice is valid?", style='readonly'),
            label="Lattice parameters" ),
            Spring(label=" ", emphasized=False, show_label=False),
        Group(
            angle_label,
            Item("sample_mount_phi", label="Sample mounting angle phi\n (1st rotation, around Y)", format_str=fmt),
            Item("sample_mount_chi", label="Sample mounting angle chi\n (2nd rotation, around Z)", format_str=fmt),
            Item("sample_mount_omega", label="Sample mounting angle omega\n (3rd rotation, around Y)", format_str=fmt),
            label="Sample mounting"),
            resizable=True
            )
            
        #Make it into a control, put it in the notebook
        self.control_manual = self.crystal.edit_traits(parent=self.panelManual, view=self.view_manual, kind='subpanel', handler=self.handler).control
        self.notebook.AddPage(self.panelManual, "Manually Enter Lattice")
        self.boxSizerManual.AddWindow(self.control_manual, 0, border=0, flag=wx.EXPAND)
        self.boxSizerManual.AddSpacer(wx.Size(8,8))
        self.boxSizerManual.AddWindow(self.buttonGenerateUB, 0, border=0, flag=wx.CENTER)
        self.panelManual.SetSizer(self.boxSizerManual)

        self.ub_orientation = SampleOrientationWhenUBMatrixWasSaved()
        self.control_load_angles = self.ub_orientation.edit_traits(parent=self.panelIsaw, kind='subpanel').control


        self._init_sizers()

    #---------------------------------------------------------------------------
    def OnbuttonOKButton(self, event):
        if not self.crystal.is_lattice_valid():
            #Can't OK when the crystal is invalid
            wx.MessageDialog(self, "Crystal is invalid! Please enter reasonable lattice parameters before clicking OK.", 'Invalid Crystal', wx.OK | wx.ICON_ERROR).ShowModal()
        else:
            #Close and apply
            self.handler.close(True)
        event.Skip()

    #---------------------------------------------------------------------------
    def OnbuttonCancelButton(self, event):
        self.handler.close(False)
        event.Skip()

    #---------------------------------------------------------------------------
    def OnButtonGenerateUB(self, event):
        self.crystal.make_ub_matrix()
        event.Skip()

    #---------------------------------------------------------------------------
    def OnButtonReadUB(self, event):
        """Ask the user to find the UB matrix file"""
        filename = self.crystal.ub_matrix_last_filename
        (path, filename) = os.path.split( os.path.abspath(filename) )
        filters = 'All files (*)|*|Text files (*.txt)|*.txt'
        dialog = wx.FileDialog ( self, defaultFile=filename, defaultDir=path, message='Choose a UB matrix file', wildcard=filters, style=wx.OPEN )
        if dialog.ShowModal() == wx.ID_OK:
            filename = dialog.GetPath()
            dialog.Destroy()
        else:
            #'Nothing was selected.
            dialog.Destroy()
            return
        #Load if it exists
        if not os.path.exists(filename):
            wx.MessageDialog(self, "File '%s' was not found!" % filename, 'Error', wx.OK | wx.ICON_ERROR).ShowModal()
        else:
            #Now load the file, using the sample orientation angles from before
            angles = np.array([self.ub_orientation.phi_degrees, self.ub_orientation.chi_degrees, self.ub_orientation.omega_degrees])
            angles = np.deg2rad(angles)
            self.crystal.read_ISAW_ubmatrix_file(filename, angles)

        event.Skip()



#----------------------------------------------------------------------
#The last dialog to be shown
last_dialog = None

def show_dialog(parent, crystal):
    """Open the dialog to edit crystal settings.

    Parameters:
        parent: parent window or frame.
        crystal: Crystal object being modified.

    Return:
        True if the user clicked OK.
    """
    dlg = DialogEditCrystal(parent, crystal)
    #Save it as a global variable (mostly for use by scripting)
    global last_dialog
    last_dialog = dlg
    #Show it
    dlg.ShowModal()
    okay = dlg.handler.user_clicked_okay
    dlg.Destroy()
    return okay




#----------------------------------------------------------------------
if __name__=="__main__":
    sys.path.insert(0, "..")
    c = Crystal("Test")
    print show_dialog(None, c)
    print c.name
    print c.ub_matrix

