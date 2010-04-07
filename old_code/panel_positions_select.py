#Boa:FramePanel:PanelPositions
"""
PanelPositions: a GUI component showing the list of calculated sample positions, and allows
the user to select which are being considered in the coverage calculation.
"""
#$Id$

import wx

import model

import display_thread


[wxID_PANELPOSITIONS, wxID_PANELPOSITIONSBUTTONDELETEPOSITION, 
 wxID_PANELPOSITIONSBUTTONUPDATELIST, wxID_PANELPOSITIONSCHECKLISTPOSITIONS, 
 wxID_PANELPOSITIONSCHECKSELECTALL, wxID_PANELPOSITIONSSTATICTEXTHEADER, 
 wxID_PANELPOSITIONSSTATICTEXTHEADERUNITS, wxID_PANELPOSITIONSSTATICTEXTHELP, 
] = [wx.NewId() for _init_ctrls in range(8)]



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class PositionsListView():
    """View/Controller for the list of positions."""
    panel = None #type PanelPositions
    
    #----------------------------------------------------------------------------------------
    def __init__(self, panel):
        """Constructor."""
        self.panel = panel
        #Subscribe to messages about the list of positions
        model.messages.subscribe(self.update_list, model.messages.MSG_POSITION_LIST_CHANGED)
        model.messages.subscribe(self.position_list_appended, model.messages.MSG_POSITION_LIST_APPENDED)
        model.messages.subscribe(self.update_selection, model.messages.MSG_POSITION_SELECTION_CHANGED)
        #Make sure the initial draw is good
        self.position_list_changed()

    #----------------------------------------------------------------------------------------
    def make_name_from_position(self, pos):
        import numpy
        s = ""
        i = 0
        for angleinfo in model.instrument.inst.angles:
            x = pos.angles[i]
            s = s + ("%8.2f  " % angleinfo.internal_to_friendly(x))
            i += 1
        return s

    #----------------------------------------------------------------------------------------
    def select_all(self, value):
        """Select or deslect all items in the list."""
        for i in range(self.panel.checkListPositions.GetCount()):
            self.panel.checkListPositions.Check(i, value)
        #Make sure the GUI updates
        self.changed()

    #----------------------------------------------------------------------------------------
    def position_list_appended(self, message):
        """Add an item at the end of the list. Called when a new position is calculated."""
        raise NotImplementedError()

    #----------------------------------------------------------------------------------------
    def delete_all(self):
        """Deletes all entries in the position list."""
        #This deletes everything in the list
        del model.instrument.inst.positions[:]
        #Make sure to clear the parameters too, by giving it an empty dict() object.
        display_thread.clear_positions_selected()
        #Update the displayed list.
        self.update_list()

    #----------------------------------------------------------------------------------------
    def update_list(self):
        """Re-do the positions list"""
        position_names = list()
        for pos in model.instrument.inst.positions:
            position_names.append( self.make_name_from_position(pos) )
        self.panel.checkListPositions.Set(items=position_names)
        #And this re-checks the boxes as needed.
        self.update_selection(None)

    #----------------------------------------------------------------------------------------
    def update_selection(self, message):
        """Updates the GUI to reflect all the selected positions in the latest parameters.
        If message is set, this is the handler for an external change in selection.
        The message.data dictionary is used instead of latestparams.
        """
        #Re-check the previously checked items.
        if message is None:
            pos_dict = display_thread.get_positions_dict()
        else:
            pos_dict = message.data

        for i in range( self.panel.checkListPositions.GetCount() ):
            #Get the position from the instruments' list.
            pos = model.instrument.inst.positions[i]
            #Check it if you find it, and it's true.
            val = False
            if pos_dict.has_key( id(pos) ):
                val = pos_dict[ id(pos) ]
            self.panel.checkListPositions.Check(i, val )

    #----------------------------------------------------------------------------------------
    def changed(self):
        """Call when the checked list changes."""
        #We generate a dictionary of object, True/False
        pos_dict = dict()
        for i in range(len(model.instrument.inst.positions)):
            posCoverage = model.instrument.inst.positions[i]
            value = self.panel.checkListPositions.IsChecked(i)
            pos_dict[ id(posCoverage)] = value 
        
        #Save it as the parameter to change
        display_thread.select_position_coverage(pos_dict)



        
        
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class PanelPositions(wx.Panel):
    """GUI to select which positions are included in the calculation,
    and to calculate new ones."""
    positionsListView = None
    
    
    def _init_coll_boxSizerBottom_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(8, 8), border=0, flag=wx.EXPAND | wx.RIGHT)
        parent.AddWindow(self.buttonUpdateList, 0, border=0, flag=0)
        parent.AddStretchSpacer(1)
        parent.AddWindow(self.buttonDeletePosition, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=wx.EXPAND | wx.RIGHT)

    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.staticTextHelp, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(4, 4), border=0, flag=0)
        parent.AddSizer(self.boxSizerCheckboxes, 0, border=0, flag=0)
        parent.AddWindow(self.staticTextHeader, 0, border=0, flag=wx.EXPAND)
        parent.AddWindow(self.staticTextHeaderUnits, 0, border=0,
              flag=wx.EXPAND)
        parent.AddWindow(self.checkListPositions, 1, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(4, 4), border=0, flag=0)
        parent.AddSizer(self.boxSizerBottom, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(4, 4), border=0, flag=0)

    def _init_coll_boxSizerCheckboxes_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.checkSelectAll, 0, border=0, flag=0)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)

        self.boxSizerCheckboxes = wx.BoxSizer(orient=wx.VERTICAL)

        self.boxSizerBottom = wx.BoxSizer(orient=wx.HORIZONTAL)

        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerCheckboxes_Items(self.boxSizerCheckboxes)
        self._init_coll_boxSizerBottom_Items(self.boxSizerBottom)

        self.SetSizer(self.boxSizerAll)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_PANELPOSITIONS, name=u'PanelPositions',
              parent=prnt, pos=wx.Point(743, 601), size=wx.Size(330, 612),
              style=wx.TAB_TRAVERSAL)
        self.SetClientSize(wx.Size(330, 612))
        self.SetBackgroundColour(wx.Colour(246, 245, 245))
        self.SetAutoLayout(True)
        self.SetBackgroundStyle(wx.BG_STYLE_SYSTEM)

        self.checkListPositions = wx.CheckListBox(choices=[],
              id=wxID_PANELPOSITIONSCHECKLISTPOSITIONS,
              name=u'checkListPositions', parent=self, pos=wx.Point(0, 92),
              size=wx.Size(330, 483), style=0)
        self.checkListPositions.SetAutoLayout(False)
        self.checkListPositions.SetMinSize(wx.Size(10, 10))
        self.checkListPositions.SetFont(wx.Font(10, 76, wx.NORMAL, wx.NORMAL,
              False, u'Monospace'))
        self.checkListPositions.Bind(wx.EVT_CHECKLISTBOX,
              self.OnCheckListPositionsChecklistbox,
              id=wxID_PANELPOSITIONSCHECKLISTPOSITIONS)

        self.staticTextHelp = wx.StaticText(id=wxID_PANELPOSITIONSSTATICTEXTHELP,
              label=u'Check the sample orientations you wish to use in the calculation:',
              name=u'staticTextHelp', parent=self, pos=wx.Point(0, 0),
              size=wx.Size(330, 32), style=0)
        self.staticTextHelp.SetMinSize(wx.Size(71, 32))

        self.staticTextHeader = wx.StaticText(id=wxID_PANELPOSITIONSSTATICTEXTHEADER,
              label=u'        Phi      Chi    Omega', name=u'staticTextHeader',
              parent=self, pos=wx.Point(0, 58), size=wx.Size(330, 17), style=0)
        self.staticTextHeader.SetFont(wx.Font(10, 76, wx.NORMAL, wx.NORMAL,
              False, u'Monospace'))

        self.checkSelectAll = wx.CheckBox(id=wxID_PANELPOSITIONSCHECKSELECTALL,
              label=u'Select All', name=u'checkSelectAll', parent=self,
              pos=wx.Point(0, 36), size=wx.Size(95, 22), style=0)
        self.checkSelectAll.SetValue(False)
        self.checkSelectAll.Bind(wx.EVT_CHECKBOX, self.OnCheckSelectAllCheckbox,
              id=wxID_PANELPOSITIONSCHECKSELECTALL)

        self.buttonDeletePosition = wx.Button(id=wxID_PANELPOSITIONSBUTTONDELETEPOSITION,
              label=u'Delete Positions', name=u'buttonDeletePosition',
              parent=self, pos=wx.Point(202, 579), size=wx.Size(120, 29),
              style=0)
        self.buttonDeletePosition.Bind(wx.EVT_BUTTON,
              self.OnButtonDeletePositionButton,
              id=wxID_PANELPOSITIONSBUTTONDELETEPOSITION)

        self.buttonUpdateList = wx.Button(id=wxID_PANELPOSITIONSBUTTONUPDATELIST,
              label=u'Refresh List', name=u'buttonUpdateList', parent=self,
              pos=wx.Point(8, 579), size=wx.Size(120, 29), style=0)
        self.buttonUpdateList.Bind(wx.EVT_BUTTON, self.OnButtonUpdateListButton,
              id=wxID_PANELPOSITIONSBUTTONUPDATELIST)

        self.staticTextHeaderUnits = wx.StaticText(id=wxID_PANELPOSITIONSSTATICTEXTHEADERUNITS,
              label=u'       (deg)', name=u'staticTextHeaderUnits', parent=self,
              pos=wx.Point(0, 75), size=wx.Size(330, 17), style=0)
        self.staticTextHeaderUnits.SetFont(wx.Font(10, 76, wx.NORMAL, wx.NORMAL,
              False, u'Monospace'))
        self.staticTextHeaderUnits.SetForegroundColour(wx.Colour(122, 122, 122))

        self._init_sizers()

    def __init__(self, parent):
        self._init_ctrls(parent)
        
        #Additional code
        self.positionsListView = PositionsListView(self)

        #Make the header label
        s = "    "
        su = s
        for angleinfo in model.instrument.inst.angles:
            s = s + "%8s  " % angleinfo.name
            su = su + "%8s  " % ( "("+angleinfo.friendly_units+")" )
        self.staticTextHeader.SetLabel(s)
        self.staticTextHeaderUnits.SetLabel(su)
        
##        
##        #Additional custom code
##        self.panelAddPositions = panel_add_positions.PanelAddPositions(
##                            id=-1, name='', parent=self.panelToHoldAddPositions,
##                            pos=-1, size=-1, style=wx.TAB_TRAVERSAL)
##        #Sizer to make sure the panel expands to the holder panel
##        sizer = wx.StaticBoxSizer(self.staticBoxNewPositions)
##        sizer.Add(self.panelAddPositions, 1, wx.EXPAND)
##        self.panelToHoldAddPositions.SetSizer(sizer)
##        

#-------------------------------------------------------------------------------
#----------------------- EVENT HANDLERS ----------------------------------------
#-------------------------------------------------------------------------------

    def OnCheckListPositionsChecklistbox(self, event):
        #The list of checked items has changed
        self.positionsListView.changed()
        event.Skip()

    def OnButtonUpdateListButton(self, event):
        self.positionsListView.update_list()
        event.Skip()

    def OnCheckSelectAllCheckbox(self, event):
        #Select or de-slect all
        value = self.checkSelectAll.GetValue()
        self.positionsListView.select_all(value)
        event.Skip()

    def OnButtonDeletePositionButton(self, event):
        ret = wx.MessageBox("Do you really want to delete all the entries in the orientations list?", "Delete All?", style=wx.YES_NO | wx.NO_DEFAULT)
        if ret == wx.YES:
            self.positionsListView.delete_all()
        event.Skip()
        
        
        

if __name__ == "__main__":
    import gui_utils
    (app, pnl) = gui_utils.test_my_gui(PanelPositions)
    app.MainLoop()
