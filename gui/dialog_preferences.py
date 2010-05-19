#Boa:Dialog:DialogPreferences
"""Simple dialog to change global program preferences."""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import CrystalPlan_version

#--- GUI Imports ---
import config_gui

#--- Model Imports ---
import model

#--- Traits imports ---
from enthought.traits.api import HasTraits,Int,Float,Str,String,Property,Bool, List, Tuple, Array
from enthought.traits.ui.api import View,Item,Group,Label,Heading, Spring, Handler, TupleEditor, TabularEditor, ArrayEditor, TextEditor, CodeEditor
from enthought.traits.ui.menu import OKButton, CancelButton,RevertButton
from enthought.traits.ui.menu import Menu, Action, Separator


def create(parent):
    return DialogPreferences(parent)

[wxID_DIALOGPREFERENCES, wxID_DIALOGPREFERENCESBUTTONCANCEL, 
 wxID_DIALOGPREFERENCESBUTTONOK, 
] = [wx.NewId() for _init_ctrls in range(3)]

class DialogPreferences(wx.Dialog):
    """Dialog to change display and calculation preferences."""
    
    def _init_coll_boxSizerButtons_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(8,8))
        parent.AddStretchSpacer(prop=1)
        parent.AddWindow(self.buttonOK, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8,8))
        parent.AddWindow(self.buttonCancel, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8,8))

    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(8,8))
        parent.AddSizer(self.boxSizerButtons, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8,8))

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)

        self.boxSizerButtons = wx.BoxSizer(orient=wx.HORIZONTAL)

        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerButtons_Items(self.boxSizerButtons)

        self.SetSizer(self.boxSizerAll)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Dialog.__init__(self, id=wxID_DIALOGPREFERENCES,
              name=u'DialogPreferences', parent=prnt, pos=wx.Point(525, 314),
              size=wx.Size(631, 703), style=wx.DEFAULT_DIALOG_STYLE,
              title=u'Change Preferences')
        self.SetClientSize(wx.Size(500, 400))
        self.SetIcon( wx.Icon(CrystalPlan_version.icon_file_config, wx.BITMAP_TYPE_PNG) )


        self.buttonOK = wx.Button(id=wxID_DIALOGPREFERENCESBUTTONOK,
              label=u'OK', name=u'buttonOK', parent=self, pos=wx.Point(0, 0),
              size=wx.Size(85, 29), style=0)
        self.buttonOK.Bind(wx.EVT_BUTTON, self.OnButtonOKButton,
              id=wxID_DIALOGPREFERENCESBUTTONOK)

        self.buttonCancel = wx.Button(id=wxID_DIALOGPREFERENCESBUTTONCANCEL,
              label=u'Cancel', name=u'buttonCancel', parent=self,
              pos=wx.Point(0, 29), size=wx.Size(85, 29), style=0)
        self.buttonCancel.Bind(wx.EVT_BUTTON, self.OnButtonCancelButton,
              id=wxID_DIALOGPREFERENCESBUTTONCANCEL)

        self._init_sizers()

    def __init__(self, parent):
        self._init_ctrls(parent)
        #Make a view combining the 2 types of configurations
        
        view= View(
            Group(
                Item('cg.show_d_spacing', label="Display in d-spacing (instead of q-space)?"),
                Item('cg.label_corners'),
                Item('cg.max_3d_points'),
                Group(Label("Note: To apply these options, you may\nneed to close and re-open windows\nsuch as the reciprocal space 3D viewer,\nor redo some calculations.")),
                label="GUI Options"),
            Group(
                Item('c.force_pure_python'),
                Item('c.use_multiprocessing'),
                Item('c.reflection_divergence_degrees'),
                Item('c.default_detector_filename'),
                label="General Options"
                 ))

        #Save the starting config values
        self.starting_config_gui = config_gui.GuiConfig()
        self.starting_config_gui.copy_traits(config_gui.cfg)
        self.starting_model_config = model.config.ModelConfig()
        self.starting_model_config.copy_traits(model.config.cfg)

        #Make it into a control
        self.handler = None
        self.control = config_gui.cfg.edit_traits(parent=self, view=view, kind='subpanel', handler=self.handler,
                        context={'cg':config_gui.cfg, 'c':model.config.cfg}).control
        self.boxSizerAll.InsertWindow(0, self.control, 1, border=8, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP)
        self.GetSizer().Layout()




    def OnButtonOKButton(self, event):
        self.Close()
        event.Skip()

    def OnButtonCancelButton(self, event):
        #Revert the configs back to what we saved
        config_gui.cfg.copy_traits(self.starting_config_gui)
        model.config.cfg.copy_traits(self.starting_model_config)
        #Close it.
        self.Close()
        event.Skip()



#--------------------------------------------------------------------
if __name__ == '__main__':
    #Test routine
    import gui_utils
    (app, frm) = gui_utils.test_my_gui(DialogPreferences)
    app.MainLoop()

