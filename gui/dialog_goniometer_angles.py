"""
GUI to set crystal parameters, as a dialog.
"""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id: dialog_edit_crystal.py 1424 2010-10-18 17:38:30Z 8oz $

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
class GoniometerEditorTraitsHandler(Handler):
    """Handler that reacts to changes in the goniometer object."""

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
        # Do nothing for now - always valid
        pass

    #---------------------------------------------------------------------------
    def close(self, apply_change):
        """Close """
        if apply_change:
            #Modify the crystal data in the experiment.
            self.frame.original_gonio.copy_traits(self.frame.gonio)
        #Close it
        self.user_clicked_okay = apply_change
        self.frame.Close()



# ===========================================================================================
class DialogGoniometerAngles(wx.Dialog):
    """Dialog to edit advanced goniometer angles"""

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

    def _init_sizers(self):
        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerIsaw_Items(self.boxSizerIsaw)
        self._init_coll_boxSizerButtons_Items(self.boxSizerButtons)
        self.SetSizer(self.boxSizerAll)


    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Dialog.__init__(self, name=u'DialogEditCrystal',
              parent=prnt, pos=wx.Point(202, 235), 
              style= wx.RESIZE_BORDER | wx.DEFAULT_DIALOG_STYLE | wx.MAXIMIZE_BOX | wx.MINIMIZE_BOX,
              title=u'Advanced Goniometer Settings Editor')
        self.SetClientSize(wx.Size(500, 620))

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


    #---------------------------------------------------------------------------
    def __init__(self, parent, gonio):
        #Basic controls
        self._init_ctrls(parent)
        
        #Setup the parameter editor traits ui panel
        self.original_gonio = gonio
        self.gonio = copy.deepcopy(gonio)
        self.handler = GoniometerEditorTraitsHandler(self)


        #--- Make all the panels ---
        self.panels = []

        #@type gonio Goniometer
        for ang in self.gonio.gonio_angles + self.gonio.wl_angles:
            panel = wx.Panel(id=wx.NewId(), name="panel", parent=self.notebook, style=wx.TAB_TRAVERSAL)
            self.notebook.AddPage(panel, ang.name, select=True)

            #Make a box sizer that will hold the traits gui thingie
            panel_boxSizer = wx.BoxSizer(orient=wx.VERTICAL)
            view = View( Group(
            Item("name", label="Angle Name")
            )
            )
            panel_control = ang.edit_traits(parent=panel, kind='subpanel').control
            panel_boxSizer.AddWindow(panel_control, 0, border=0, flag=wx.EXPAND)
            panel.SetSizer(panel_boxSizer)

            #Save it for later
            self.panels.append(panel)

        
        #Make all the sizers
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerIsaw = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerManual = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerButtons = wx.BoxSizer(orient=wx.HORIZONTAL)

        #Make the TRAITS view

        #At the top is a general view
        self.view_top =  View( Group(
            Item("name", label="Goniometer Name"),
            Label("Use the tabs below to change advanced\nsettings about the goniometer\nangles and wavelength control.")
            )
            )
        #Make the control and add it
        self.control_top = self.gonio.edit_traits(parent=self, view=self.view_top, kind='subpanel', handler=self.handler).control
        self.boxSizerAll.AddWindow(self.control_top, 0, border=0, flag=wx.EXPAND)

        # Now add the notebook, make it expand
        self.boxSizerAll.AddWindow(self.notebook, 1, border=4, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP)

        #self.notebook.AddPage(self.panelManual, "Manually Enter Lattice")



        self._init_sizers()

    #---------------------------------------------------------------------------
    def OnbuttonOKButton(self, event):
        #Close and apply
        self.handler.close(True)
        event.Skip()

    #---------------------------------------------------------------------------
    def OnbuttonCancelButton(self, event):
        self.handler.close(False)
        event.Skip()



#----------------------------------------------------------------------
#The last dialog to be shown
last_dialog = None

def show_dialog(parent, gonio):
    """Open the dialog to edit a goniometer advanced angles

    Parameters:
        parent: parent window or frame.
        gonio: Goniometer to change

    Return:
        True if the user clicked OK.
    """
    dlg = DialogGoniometerAngles(parent, gonio)
    #Save it as a global variable (mostly for use by scripting)
    global last_dialog
    last_dialog = dlg
    #Show it
    dlg.ShowModal()
    okay = dlg.handler.user_clicked_okay
    dlg.Destroy()
    return okay



#====================================================================================
if __name__ == "__main__":
    model.instrument.inst = model.instrument.Instrument()
    model.goniometer.initialize_goniometers()

    g = model.goniometer.TopazInHouseGoniometer()
    print show_dialog(None, g)
    print g.name
