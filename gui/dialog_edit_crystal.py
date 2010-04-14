"""
GUI to set crystal parameters, as a dialog.
"""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import os
import copy

#--- Traits imports ---
from enthought.traits.api import HasTraits,Int,Float,Str,String,Property,Bool, List, Tuple, Array
from enthought.traits.ui.api import View,Item,Group,Label,Heading, Spring, Handler, TupleEditor, TabularEditor, ArrayEditor, TextEditor, CodeEditor
from enthought.traits.ui.editors import EnumEditor

#--- GUI Imports ---

#--- Model Imports ---
import model
from model.crystals import Crystal

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
        #Do we change the ub matrix?
        if self.frame.crystal.generate_ub_matrix:
            self.frame.crystal.make_ub_matrix()
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
        # generated method, don't edit

        parent.AddWindow(self.staticTextSpacer1, 1, border=0, flag=0)
        parent.AddWindow(self.buttonOK, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)
        parent.AddWindow(self.buttonCancel, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)

    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit
        parent.AddSizer(self.boxSizerParams, 1, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)
        parent.AddWindow(self.buttonReadUB, 0, border=0, flag=wx.CENTER)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)
        parent.AddSizer(self.boxSizerButtons, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(16, 8), border=0, flag=0)

    def _init_sizers(self):
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerParams = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerButtons = wx.BoxSizer(orient=wx.HORIZONTAL)
        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerButtons_Items(self.boxSizerButtons)

        self.SetSizer(self.boxSizerAll)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Dialog.__init__(self, name=u'DialogEditCrystal',
              parent=prnt, pos=wx.Point(702, 235), size=wx.Size(475, 600),
              style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
              title=u'Edit Crystal Parameters')
        self.SetClientSize(wx.Size(500, 750))

        self.buttonReadUB = wx.Button(
              label=u'Read UB matrix from file...', name=u'buttonReadUB', parent=self,
              pos=wx.Point(16, 563), size=wx.Size(250, 29), style=0)
        self.buttonReadUB.Bind(wx.EVT_BUTTON, self.OnButtonReadUB)

        self.buttonOK = wx.Button(
              label=u'Ok', name=u'buttonOK', parent=self,
              pos=wx.Point(16, 563), size=wx.Size(150, 29), style=0)
        self.buttonOK.Bind(wx.EVT_BUTTON, self.OnbuttonOKButton)

        self.buttonCancel = wx.Button(
              label=u'Cancel', name=u'buttonCancel', parent=self, pos=wx.Point(309,
              563), size=wx.Size(150, 29), style=0)
        self.buttonCancel.Bind(wx.EVT_BUTTON, self.OnbuttonCancelButton)

        self.staticTextSpacer1 = wx.StaticText(label=u' ', name=u'staticTextSpacer1', parent=self,
              pos=wx.Point(166, 563), size=wx.Size(320, 17), style=0)

        self._init_sizers()

    #---------------------------------------------------------------------------
    def __init__(self, parent, my_crystal):
        self._init_ctrls(parent)

        #Make the TRAITS view
        angle_label = Group(
                    Label(label="""If all the sample mounting angles are zero, the sample's crystal 
lattice coordinate system is aligned with the instrument coordinates.
The 'a' vector is parallel to x; 'b' is in the XY plane towards +y;
'c' goes towards +z.
""",
                    emphasized=False, show_label=True),
                    )

        fmt = "%.3f"
        view = View(
        Group(
            Item("name", label="Crystal Name"),
            Item("description", label="Description:", editor=TextEditor(multi_line=True)),
            label="Crystal info" ),
            Spring(label=" ", emphasized=False, show_label=False),
        Group(
            Item("lattice_lengths_arr", label="Lattice dimensions\n(a,b,c - in Angstroms)", format_str=fmt),
            Item("lattice_angles_deg_arr", label="Lattice angles\n(alpha,beta,gamma - in degrees)", format_str=fmt),
            Item("valid_parameters_yesno", label="Lattice is valid?", style='readonly'),
            Item("point_group_name", label="Point Group:", editor=EnumEditor(name="point_group_name_list")),
            label="Lattice parameters" ),
            Spring(label=" ", emphasized=False, show_label=False),
        Group(
            angle_label,
            Item("sample_mount_phi", label="Sample mounting angle phi\n (1st rotation, around Y)", format_str=fmt, enabled_when="generate_ub_matrix"),
            Item("sample_mount_chi", label="Sample mounting angle chi\n (2nd rotation, around Z)", format_str=fmt, enabled_when="generate_ub_matrix"),
            Item("sample_mount_omega", label="Sample mounting angle omega\n (3rd rotation, around Y)", format_str=fmt, enabled_when="generate_ub_matrix"),
            label="Sample mounting"),

            Spring(label=" ", emphasized=False, show_label=False),
            Item("generate_ub_matrix", label="Generate UB matrix using sample mount angles?\nSample mounting is ignored unless checked."),
        Group(
            Item("ub_matrix", label="Sample's UB Matrix", style='readonly', format_str="%9.5f"),
            ),
            resizable=True
            )
        #Setup the parameter editor traits ui panel
        self.original_crystal = my_crystal
        self.crystal = copy.copy(my_crystal)
        self.handler = CrystalEditorTraitsHandler(self)
        #Make it into a control
        self.control = self.crystal.edit_traits(parent=self, view=view, kind='subpanel', handler=self.handler).control
        self.boxSizerParams.AddWindow(self.control, 3, border=1, flag=wx.EXPAND)
        self.GetSizer().Layout()


    #---------------------------------------------------------------------------
    def OnbuttonOKButton(self, event):
        if not self.crystal.is_lattice_valid():
            #Can't OK when the crystal is invalid
            wx.MessageDialog(self, "Crystal is invalid! Please enter reasonable lattice parameters before clicking OK.", 'Invalid Crystal', wx.OK | wx.ICON_ERROR).ShowModal()
        else:
            #Close and apply
            self.handler.close(True)
        event.Skip()

    def OnbuttonCancelButton(self, event):
        self.handler.close(False)
        event.Skip()

    #---------------------------------------------------------------------------
    def OnButtonReadUB(self, event):
        """Ask the user to find the UB matrix file"""
        filename = self.crystal.ub_matrix_last_filename
        (path, ignored) = os.path.split( os.path.abspath(filename) )
        filters = 'All files (*.*)|*.*|Text files (*.txt)|*.txt'
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
            self.crystal.read_ubmatrix_file(filename)

        event.Skip()



#----------------------------------------------------------------------
def show_dialog(parent, crystal):
    """Open the dialog to edit crystal settings.

    Parameters:
        parent: parent window or frame.
        crystal: Crystal object being modified.

    Return:
        True if the user clicked OK.
    """
    dlg = DialogEditCrystal(parent, crystal)
    dlg.ShowModal()
    okay = dlg.handler.user_clicked_okay
    dlg.Destroy()
    return okay




#----------------------------------------------------------------------
if __name__=="__main__":
    c = Crystal("Test")
    print show_dialog(None, c)
    print c.name
    print c.ub_matrix

