#Boa:Frame:FrameQspaceView
"""Frame to view reciprocal space coverage."""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import time
import numpy as np

#--- GUI Imports ---
import panel_qspace_options
import panel_coverage_stats
import panel_reflections_view_options
import frame_reflection_info
import config_gui
import display_thread

#--- Model Imports ---
import model
from model.numpy_utils import column, vector_length

#--- Mayavi Imports ---
try:
    from enthought.traits.api import HasTraits, Instance
    from enthought.traits.ui.api import View, Item
    from enthought.mayavi.sources.api import ArraySource
    from enthought.mayavi.modules.iso_surface import IsoSurface
    from enthought.mayavi.modules.api import Volume, Surface
    from enthought.mayavi.filters.contour import Contour
    from enthought.mayavi.filters.api import PolyDataNormals
    from enthought.mayavi.filters.set_active_attribute import SetActiveAttribute

    from enthought.mayavi.modules.outline import Outline
    from enthought.mayavi.modules.grid_plane import GridPlane
    from enthought.mayavi.modules.contour_grid_plane import ContourGridPlane
    from enthought.mayavi.modules.scalar_cut_plane import ScalarCutPlane
    from enthought.mayavi.api import Engine
    from enthought.tvtk.pyface.scene_editor import SceneEditor
    from enthought.mayavi.tools.mlab_scene_model import MlabSceneModel
    import enthought.mayavi.mlab as mlab

    from enthought.tvtk.api import tvtk
    from enthought.mayavi.sources.vtk_data_source import VTKDataSource
    from enthought.mayavi.modules.api import Outline, Surface, Glyph, Text

except ImportError, e:
    print "FrameQspaceView: ERROR IMPORTING MAYAVI MODULES - 3D WILL NOT WORK!"
    



#-------------------------------------------------------------------------------
#------ SINGLETON --------------------------------------------------------------
#-------------------------------------------------------------------------------

_instance = None

def create(parent):
    global _instance
    print "This is FrameQspaceView, creating a new instance of it"
    _instance = FrameQspaceView(parent)
    return _instance

def get_instance(parent):
    """Returns the singleton instance of this frame (window)."""
    global _instance
    if _instance is None:
        return create(parent)
    else:
        #TODO: Bring to front.
##        coverage.application.SetTopWindow(_instance)
##        _instance.
        return _instance

    

#-------------------------------------------------------------------------------
#-------- VIEW/CONTROLLER ------------------------------------------------------
#-------------------------------------------------------------------------------
class QspaceViewController(HasTraits):
    """MayaVi Scene View, for showing the q-space visualization."""

    # Scene being displayed.
    scene = Instance(MlabSceneModel, ())
    
    # The layout of the panel created by Traits
    view = View(Item('scene', editor=SceneEditor(), resizable=True,
                    show_label=False), resizable=True)
                    
    #The FrameQspaceView that is calling this.
    parent_frame = None

    #Text to show when mouse is not over a reflection
    MOUSE_TEXT_WITH_NO_REFLECTION = "hkl: N/A"

    #Scaling applied to sphere from the given size parameter
    SPHERE_SCALING = 0.05

    #Reflections shown as pixels
    pixel_view = False

    #-----------------------------------------------------------------------------------------------
    def __init__(self, parent_frame):
        #Initialize the parent class
        HasTraits.__init__(self)

        #Record the frame calling this
        self.parent_frame = parent_frame

        #And more initialization
        self.warning_text_visible = False
        self.init_view_objects()

        #Subscribe to messages
        model.messages.subscribe(self.update_stats_panel, model.messages.MSG_EXPERIMENT_QSPACE_CHANGED)
        model.messages.subscribe(self.update_data_volume, model.messages.MSG_EXPERIMENT_QSPACE_CHANGED)
        model.messages.subscribe(self.update_data_points, model.messages.MSG_EXPERIMENT_REFLECTIONS_CHANGED)
        model.messages.subscribe(self.update_stats_panel, model.messages.MSG_EXPERIMENT_REFLECTIONS_CHANGED)
        model.messages.subscribe(self.init_view_objects, model.messages.MSG_EXPERIMENT_QSPACE_SETTINGS_CHANGED)

        #Do an initial data update
        self.update_stats_panel()
        self.update_data_volume()
        self.update_data_points()
        self.view_mode_changed()
        
        #Don't open a child frame at first
        self.reflection_info_child_frame = None

    #-----------------------------------------------------------------------------------------------
    def __del__(self):
        print "QspaceViewController.__del__"
        self.cleanup()

    #-----------------------------------------------------------------------------------------------
    def cleanup(self):
        """Clean-up routine for closing the view."""
        print "QspaceViewController.cleanup"
        model.messages.unsubscribe(self.update_stats_panel)
        model.messages.unsubscribe(self.update_data_volume)
        model.messages.unsubscribe(self.update_data_points)
        model.messages.unsubscribe(self.init_view_objects)
        #Also close child windows
        if not self.reflection_info_child_frame is None:
            self.reflection_info_child_frame.Destroy()

    #-----------------------------------------------------------------------------------------------
    def init_pickers(self):
        """Initialize the pickers objects."""
        #This vtk picker object will be used later
        self.pointpicker = tvtk.PointPicker()

        #TODO: Fix it!
#        self.scene.picker.pointpicker.add_observer('EndPickEvent', picker_callback)
#        self.reflection_picker = ReflectionPicker(self.scene.picker.pointpicker)
        self.scene.interactor.add_observer('RightButtonPressEvent', self.on_button_press)
        self.scene.interactor.add_observer('MouseMoveEvent', self.on_mouse_move)
        self.scene.interactor.add_observer('RightButtonReleaseEvent', self.on_button_release)


    #-----------------------------------------------------------------------------------------------
    def get_reflection_under_mouse(self, obj, verbose=False):
        """Return the Reflection object under the mouse cursor.
        Parameters:
            obj: the object passed to the mouse observer (contains the mouse coordinates)."""

        #Directly create a VTK point picker. This avoids the mouse cursor going "wait"
        # when simply requesting a point.
        pp = self.pointpicker
        x, y = obj.GetEventPosition()
        pp.pick((float(x), float(y), 0.0), self.scene.renderer)
        
        #These are the closest coordinates found
        coordinates = pp.pick_position

        #Lets go through all the actors found
        for (i, act) in enumerate(pp.actors):
            if id(act) == id(self.points_module_surface.actor.actor):
                #Yay, we found a point on that actor - the points surface
                #That way, we ignore any elements in front
                coordinates = pp.picked_positions[i]
                break
            if id(act) == id(self.points_module_glyph.actor.actor):
                #If in glyph mode, we can use that one instead.
                # But the points surface is better.
                coordinates = pp.picked_positions[i]
                
        #Look for the closest reflection to those coordinates
        return model.experiment.exp.get_reflection_closest_to_q( column(coordinates) )

#        point_id = pp.point_id
#        return model.experiment.exp.get_reflection_from_masked_id(point_id)
    
#        if verbose:
#            actors = pp.actors
#            print len(actors)
#            for (i, act) in enumerate(actors):
#                print "actor", id(act), " position", pp.picked_positions[i]
#                if id(act) == id(self.mouse_cube.actor.actor):
#                    print "... mousecube"
#                if id(act) == id(self.outline.actor.actor):
#                    print "... outline"
#                if id(act) == id(self.points_module_surface.actor.actor):
#                    print "... reflection pixels"
                
#        if self.pixel_view:
#            #In pixel view, the point_id corresponds to the point in the reflections array (after masking)
#            #This gets the reflection from that id
#            return model.experiment.exp.get_reflection_from_masked_id(point_id)
#
#        else:
#            #--- We are in sphere mode; pain ITA! --
#            coordinates = pp.pick_position
#            #This function will look for the closest one among all the points.
#            return model.experiment.exp.get_reflection_closest_to_q( column(coordinates) )



    #-----------------------------------------------------------------------------------------------
    def on_button_press(self, obj, evt):
        """Event handler for the 3d Scene, called when right-mouse-button is clicked."""
        self.mouse_mvt = False
        refl = self.get_reflection_under_mouse(obj, True)
        #Open the frame_reflection_info (if needed) and set it to the reflection we clicked
        frm = self.open_frame_reflection_info()
        frm.Show()
        frm.panel.set_reflection(refl, update_textboxes=True)
        #This'll move the cube
        self.on_user_changing_reflection_selected(refl)

    #-----------------------------------------------------------------------------------------------
    def open_frame_reflection_info(self):
        """Open the frame_reflection_info frame if not already open.
        Makes sure that it is initialized correctly."""
        if self.reflection_info_child_frame is None:
            #Make a new one
            frm = frame_reflection_info.FrameReflectionInfo(self.parent_frame, can_follow=True, do_follow=True)
            #Also make sure we observe changes
            frm.panel.add_observer(self.on_user_changing_reflection_selected)
            frm.Bind(wx.EVT_CLOSE, self.on_reflection_info_child_frame_close)
            #Save the frame for when we close the frame
            self.reflection_info_child_frame = frm
        else:
            #Get an existing instance
            frm = self.reflection_info_child_frame

        return frm

    #-----------------------------------------------------------------------------------------------
    def on_reflection_info_child_frame_close(self, event):
        """Called when a child FrameReflectionInfo is closed."""
        #Clear the instance
        self.reflection_info_child_frame = None
        event.Skip()

    #-----------------------------------------------------------------------------------------------
    def on_user_changing_reflection_selected(self, refl):
        """Called when the user changes the HKL of the highlighted peak
        in this or another window.

        Parameter:
            refl: the newly-selected reflection
        """
        self.refl = refl
        if not refl is None:
            #Move the little cube
            self.mouse_point_data_src.data = self.make_single_point_data( refl.q_vector )
            self.mouse_cube.visible = True
        else:
            #hide the cube
            self.mouse_cube.visible = False

    #-----------------------------------------------------------------------------------------------
    def on_mouse_move(self, obj, evt):
        """Event handler for the 3d Scene."""
        self.mouse_mvt = True
        ref = self.get_reflection_under_mouse(obj)
        if ref is None:
            text = self.MOUSE_TEXT_WITH_NO_REFLECTION
        else:
            text = "hkl: %d,%d,%d" % ref.hkl
        #Show it in the little status bar.
        self.parent_frame.staticTextMouseInfo.SetLabel("Mouse is over: " + text)

        # If you change a 3D view element with every movement, the
        #  mouse switches to an hourglass each time - annoying!
#        self.mouse_text.text = text


    #-----------------------------------------------------------------------------------------------
    def on_button_release(self, obj, evt):
        """Event handler for the 3d Scene."""
        pass
    
        
    #-----------------------------------------------------------------------------------------------
    def init_view_objects(self, *args):
        """Initialize (or re-initialize) all the view elements in the scene.
        This needs to be called when q-space modeled area changes (changing the outline) or
            q-resolution, etc.
        """
        #Prevent drawing all the intermediate steps
        self.scene.disable_render = True

        #First, we need to remove any data sources and modules from a previous run,
        #   if they exist.
        # (this is only needed when chaning q-space parameters)
        for x in ['data_src', 'point_data_src', 'iso', 'points_module_surface',
                'points_module_glyph', 'outline', 'mouse_text', 'mouse_cube']:
            if hasattr(self, x):
                getattr(self, x).remove()

        #We get the qspace_displayed array from experiment and make a copy of it.
        #   This object will REMAIN here and just have its data updated.
        self.data_src = ArraySource(scalar_data = model.experiment.exp.get_qspace_displayed().copy() )
        self.data_src.scalar_name = "coverage"
        self.data_src.visible = False
        self.scene.engine.add_source(self.data_src)

        # --- Text overlay for warnings ----
        txt = Text(text="(Points were thinned down)", position_in_3d=False)
        txt.x_position=0.02
        txt.y_position=0.98
        txt.actor.text_scale_mode = "none"
        txt.actor.text_property.font_size = 14
        txt.actor.text_property.vertical_justification = "top"
        self.warning_text = txt
        self.scene.engine.add_module(self.warning_text)
        self.warning_text.visible = self.warning_text_visible
        
        #---- Make the isosurface object that goes with the volume coverage data -----
        iso = IsoSurface()
        self.iso = iso
        #Scale the isosurface so that the size is actually q-vector
        iso.actor.actor.scale = tuple(np.array([1.0, 1.0, 1.0]) * model.experiment.exp.inst.q_resolution)
        #And we offset the position so that the center is at q-space (0,0,0)
        iso.actor.actor.position = tuple(np.array([1.0, 1.0, 1.0]) * -model.experiment.exp.inst.qlim)
        #This helps them to show up right in partially transparent mode.
        iso.actor.property.backface_culling = 1
        iso.actor.property.frontface_culling = 1
        #Add the module to the data source, to make it plot that data
        self.data_src.add_module(iso)

        # ---- Now we make a point data source, for the individual reflection plot ---
        self.point_data_src = VTKDataSource(name="Reflection point positions")
        self.point_data_src.visible = False
        self.scene.engine.add_source(self.point_data_src)
        self.point_data_src.data = self.make_point_data() #still needs to get set.

        # ---- Make a module of simple points, using the Surface module ----
        self.points_module_surface = Surface()
        self.points_module_surface.name = "Single reflections as pixels (Surface)"
        self.point_data_src.add_module(self.points_module_surface)
        # points representation = just plot a pixel for each vertex.
        self.points_module_surface.actor.property.set(representation = 'points', point_size = 3)

        # ---- Make a module of glyphs, making spheres for each point ----
        self.points_module_glyph = Glyph()
        self.points_module_glyph.name = "Single reflections as spheres (Glyph)"
        #No scaling of glyph size with scalar data
        self.points_module_glyph.glyph.scale_mode = 'data_scaling_off'
        gs = self.points_module_glyph.glyph.glyph_source
        gs.glyph_source = gs.glyph_dict['sphere_source']
        #How many vertices does each sphere have? 3 = fastest.
        gs.glyph_source.phi_resolution = 3
        gs.glyph_source.theta_resolution = 3
        #And how big does each sphere end up?
        gs.glyph_source.radius = 0.1
        #Add the module to the same point data source, to show it
        self.point_data_src.add_module(self.points_module_glyph)

        # Hide points initially
        self.points_module_surface.visible = False
        self.points_module_glyph.visible = False


        # ---- Simple outline for all of the data. -----
        self.outline = Outline()
        #Manually make the outline = the modeled volume, which is +- qlim = 1/d_min
        self.outline.manual_bounds = True
        self.outline.bounds = tuple(np.array([-1.,1.,-1.,1.,-1.,1.]) * model.experiment.exp.inst.qlim)
        #Add it to the scene directly.
        self.scene.engine.add_module(self.outline)

        # ---- A text overlay, to show what is under the mouse -----
#        self.mouse_text = Text(text=self.MOUSE_TEXT_WITH_NO_REFLECTION, position_in_3d=False)
#        self.mouse_text.x_position=0.02
#        self.mouse_text.y_position=0.98
#        self.mouse_text.actor.text_scale_mode = "none"
#        self.mouse_text.actor.text_property.font_size = 20
#        self.mouse_text.actor.text_property.vertical_justification = "top"
#        self.scene.engine.add_module(self.mouse_text)


        # ---- A cube highlighting where the mouse is ----
        self.mouse_cube = Glyph()
        self.mouse_cube.name = "Cube highlighting mouse position (Glyph)"
        #No scaling of glyph size with scalar data
        self.mouse_cube.glyph.scale_mode = 'data_scaling_off'
        self.mouse_cube.glyph.color_mode = 'no_coloring'
        gs = self.mouse_cube.glyph.glyph_source
        gs.glyph_source = gs.glyph_dict['cube_source']
        self.mouse_cube.actor.property.representation = "wireframe"
        self.mouse_cube.actor.property.specular = 1.0 # to make edge always white
        self.mouse_point_data_src = VTKDataSource()
        self.mouse_point_data_src.name = "Mouse position (VTK Data)"
        self.mouse_point_data_src.visible = True
        self.scene.engine.add_source(self.mouse_point_data_src)
        self.mouse_point_data_src.data = self.make_single_point_data( (0,0,0))
        self.mouse_point_data_src.add_module(self.mouse_cube)
        self.mouse_cube.visible = False

        #Re-enable drawing
        self.scene.disable_render = False


    #-----------------------------------------------------------------------------------------------
    def show_pipeline(self):
        """Show the Mayavi pipeline editor window."""
        mlab.show_pipeline()
        

    #-----------------------------------------------------------------------------------------------
    def make_point_data(self):
        """Generate the point data object (type tvtk.PolyData) that is needed by the
        VTKDataSource to display the points (reflections)."""
        pd = tvtk.PolyData()
        ref_q = model.experiment.exp.reflections_q_vector
        mask = model.experiment.exp.reflections_mask

        if not isinstance(ref_q, np.ndarray):
            #No array = return an empty data set
            return pd

        #How many of the points are kept?
        num = np.sum(mask)

        #Are there more points than the configuration allows?
        num_wanted = config_gui.cfg.max_3d_points
        if num > num_wanted:
            #Adjust the mask to show only so many points
            indices = np.nonzero(mask)[0]
            #Randomize list of indices
            np.random.shuffle(indices)
            #Keep only the # wanted
            indices = indices[0:num_wanted]
            #Re-generate a smaller mask
            mask = mask.copy() #Must copy the array, otherwise we are modifying the source one
            mask &= 0
            mask[indices] = True
            #Display a warning
            self.warning_text.text = "(Points thinned down from %d to %d)" % (num, num_wanted)
            self.warning_text_visible = True
            #Make sure the number is correct
            num = num_wanted
        else:
            #No problem
            self.warning_text_visible = False
        self.warning_text.visible = self.warning_text_visible


        #Positions = our array of q-vectors.
        # .T = transpose
        pd.points = ref_q[:, mask].T

        #Create the vertices
        verts = np.arange(0, num, 1)
        verts.shape = (num, 1)
        pd.verts = verts
        #Put the # of times measured here as the scalar data
        if display_thread.get_reflection_masking_params().primary_reflections_only:
            #SHow times measured COUNTING equivalent (symmetrical) reflections.
            pd.point_data.scalars = model.experiment.exp.reflections_times_measured_with_equivalents[mask, :]
        else:
            # Show only the times measured for the exact reflection
            pd.point_data.scalars = model.experiment.exp.reflections_times_measured[mask, :]
        pd.point_data.scalars.name = 'scalars'

        return pd

    #-----------------------------------------------------------------------------------------------
    def make_single_point_data(self, coordinates):
        """Generate a pointdata object with a single point"""
        pd = tvtk.PolyData()
        #Create the vertices
        num = 1
        pd.points = np.array(coordinates).reshape(1,3)
        verts = np.arange(0, num, 1)
        verts.shape = (num, 1)
        pd.verts = verts
        pd.point_data.scalars = np.array([1])
        pd.point_data.scalars.name = 'scalars'
        return pd


    #-----------------------------------------------------------------------------------------------
    def fix_colorscale(self, module):
        """Fix the color scale of the given module, to make sure
        all visual components have matching colors.
        """
        lutm = module.module_manager.scalar_lut_manager
        lutm.data_range = np.array([0.0, 4.5])
        lutm.number_of_labels = 5
        lutm.use_default_range = False
        lutm.shadow = True
        lutm.scalar_bar.title = "Redundancy"


    #-----------------------------------------------------------------------------------------------
    def auto_adjust_pixel_size(self):
        """Change the size of the pixels drawn to make them look good on screen.
        Call this when the window is resized or when the # of points plotted changes."""
        #Figure out a good pixel size
        (w, h) = self.scene.render_window.size
        diag = np.sqrt(w*w + h*h)
        display = display_thread.get_reflection_display_params()
        pixel_size = diag/300.# * (display.size / 5.)
        self.points_module_surface.actor.property.set(representation = 'points', point_size = pixel_size)
        #Update the GUI
        self.parent_frame.tabReflections.change_displayed_size(pixel_size)

    #-----------------------------------------------------------------------------------------------
    def get_best_sphere_size(self):
        """Calculate the best size a sphere should be drawn to be spaced out okay."""
        #Calculate the spacing between hkls in reciprocal space
        a = vector_length(model.experiment.exp.crystal.recip_a)
        b = vector_length(model.experiment.exp.crystal.recip_b)
        c = vector_length(model.experiment.exp.crystal.recip_c)
        #Take a fraction of the smallest distance
        return np.min([a,b,c]) / 6

    #-----------------------------------------------------------------------------------------------
    def auto_adjust_sphere_size(self):
        """Change the size of the sphere glyphs drawn to make them spaced out okay.
        Call this when the density of points plotted changes."""
        sphere_size = self.get_best_sphere_size()
        #Actually change the size
        self.points_module_glyph.glyph.glyph_source.glyph_source.radius = sphere_size
        #Resize the highlighting
        self.resize_highlighting_cubes(sphere_size)
        #Update the GUI too
        self.parent_frame.tabReflections.change_displayed_size(sphere_size / self.SPHERE_SCALING)


    #-----------------------------------------------------------------------------------------------
    def resize_highlighting_cubes(self, sphere_size):
        """Resize the highlighting cubes to be proportional to the the specified sphere size."""
        #Also resize the highlighting cubes
        cube_size = 4. * sphere_size
        self.mouse_cube.glyph.glyph_source.glyph_source.x_length = cube_size
        self.mouse_cube.glyph.glyph_source.glyph_source.y_length = cube_size
        self.mouse_cube.glyph.glyph_source.glyph_source.z_length = cube_size


    #-----------------------------------------------------------------------------------------------
    def update_data_points(self, *args):
        """Called when a message is received saying that the q-space
        calculation has changed.
        Will update the graphical display. TO SHOW POINTS!!
        """
        #Do we need to adjust which elements are visible?
        change_visibility = not self.in_volume_mode()
            
        #No GUI updates while we work
        self.scene.disable_render = True

        if change_visibility:
            #Hide the iso surface
            self.iso.visible = False
        #Ensure visibility of warning text
        self.warning_text.visible = self.warning_text_visible

        #Get the display parameters last saved.
        # @type display ParamReflectionDisplay
        display = display_thread.get_reflection_display_params()

        #Generate a new data object to make
        self.point_data_src.data = self.make_point_data()

        if self.in_pixel_mode():
            # ------------ Pixel View -----------------------
            self.pixel_view = True
            
            if change_visibility:
                #Hide the sphere view, show the points
                self.points_module_glyph.visible = False
                self.points_module_surface.visible = True

            #Change the pixel size
            if display.automatic_size:
                self.auto_adjust_pixel_size()
            else:
                #Directly use the value
                pixel_size = round(display.size)
                self.points_module_surface.actor.property.set(representation = 'points', point_size = pixel_size)

            #Make sure the highlight cubes are okay sized
            self.resize_highlighting_cubes(self.get_best_sphere_size())

        else:
            # ------------ Sphere View -----------------------
            self.pixel_view = False
            
            if change_visibility:
                #Hide the points view, show the spheres
                self.points_module_glyph.visible = True
                self.points_module_surface.visible = False

            #Change the radius of the spheres
            if display.automatic_size:
                self.auto_adjust_sphere_size()
            else:
                #Use the number given
                sphere_size = display.size * self.SPHERE_SCALING
                self.points_module_glyph.glyph.glyph_source.glyph_source.radius = sphere_size
                self.resize_highlighting_cubes(sphere_size)

        # Doc says: Call this function when you change the array data in-place.
        #   This call should update the MayaVi views.
        # NOTE! Calling this from a different thread seems to crash.
        self.point_data_src.update()

        #Make sure the color scale is good
        self.fix_colorscale(self.points_module_surface)

        #This will redraw now.
        self.scene.disable_render = False
        
        self.point_data_needs_updating = False

        #The slice panel updates separately since it is subscribed to the messages.


    #-----------------------------------------------------------------------------------------------
    def update_data_volume(self, *args):
        """Called when a message is received saying that the q-space
        calculation has changed.
        Will update the graphical display."""

        self.scene.disable_render = True

        #Get the display parameters last saved.
        display = display_thread.get_display_params()

        #We make a copy of the data into the existing object.
        self.data_src.scalar_data = model.experiment.exp.get_qspace_displayed().copy()

        #The max coverage is necessary to avoid going too high in the isosurfaces
        if display.show_redundancy:
            max_coverage = np.max(model.experiment.exp.get_qspace_displayed())
            if max_coverage > 4: max_coverage = 4

        #Update the contours
        iso = self.iso
        if display.show_redundancy:
            iso.contour.contours = list(np.arange(1, max_coverage+1))
            iso.actor.property.opacity = 0.6
        else:
            iso.contour.contours = [1.0]
            iso.actor.property.opacity = 1.0

        #Don't use opacity in inverted mode
        if display_thread.is_inverted():
            iso.actor.property.opacity = 1.0

        iso.actor.mapper.scalar_visibility = 1 #1 means use the look-up table

        #Make sure the color scale is good
        self.fix_colorscale(self.iso)
        
        # Doc says: Call this function when you change the array data in-place.
        #   This call should update the MayaVi views.
        # NOTE! Calling this from a different thread seems to crash.
        self.data_src.update()
        
        self.scene.disable_render = False

        self.volume_data_needs_updating = False

        #The slice panel updates separately since it is subscribed to the messages.


    #-----------------------------------------------------------------------------------------------
    def in_volume_mode(self):
        """Return True if the "volume coverage" view mode selected.
        """
        tab_selected = self.parent_frame.notebookView.GetPage(self.parent_frame.notebookView.GetSelection())
        return (tab_selected is self.parent_frame.tabVolume)

    #-----------------------------------------------------------------------------------------------
    def in_pixel_mode(self):
        """Return True if we are in pixel point view mode."""
        # @type display ParamReflectionDisplay
        display = display_thread.get_reflection_display_params()
        return display.display_as == display.DISPLAY_AS_PIXELS


    #-----------------------------------------------------------------------------------------------
    def view_mode_changed(self):
        """Called when the view mode switches from volume to points, or vice-versa."""
        if self.in_volume_mode():
            #Volume view
            self.iso.visible = True
            self.warning_text.visible = False #No need to show this warning
            self.points_module_surface.visible = False
            self.points_module_glyph.visible = False
        else:
            #Just switch the view, this goes quick!
            self.iso.visible = False
            self.warning_text.visible = self.warning_text_visible
            self.points_module_surface.visible = self.in_pixel_mode()
            self.points_module_glyph.visible = not self.in_pixel_mode()
        #And the stats need to update
        self.update_stats_panel()



    #-----------------------------------------------------------------------------------------------
    def update_stats_panel(self, *args):
        """Update the information displayed on the statistics panel."""
        exp = model.experiment.exp
        if self.in_volume_mode():
            #Volume coverage
            self.parent_frame.stats_panel.show_stats( \
                exp.use_hemisphere(),
                exp.overall_coverage,
                exp.overall_redundancy )
        else:
            #Point mode
            use_symmetry = display_thread.get_reflection_masking_params().primary_reflections_only
            if use_symmetry:
                self.parent_frame.stats_panel.show_reflection_stats(use_symmetry, exp.reflection_stats_with_symmetry)
            else:
                self.parent_frame.stats_panel.show_reflection_stats(use_symmetry, exp.reflection_stats)
        
        

[wxID_FRAMEQSPACEVIEW, wxID_FRAMEQSPACEVIEWBUTTONADVANCEDVIEW, 
 wxID_FRAMEQSPACEVIEWGAUGECOVERAGE, wxID_FRAMEQSPACEVIEWGAUGEREDUNDANCY, 
 wxID_FRAMEQSPACEVIEWPANEL3D, wxID_FRAMEQSPACEVIEWPANELBOTTOM, 
 wxID_FRAMEQSPACEVIEWPANELSTATS, wxID_FRAMEQSPACEVIEWPANEL_TO_HOLD_SLICE, 
 wxID_FRAMEQSPACEVIEWSPLITTERALL, wxID_FRAMEQSPACEVIEWSTATICLINESPACER, 
 wxID_FRAMEQSPACEVIEWSTATICTEXTSTATS1, 
 wxID_FRAMEQSPACEVIEWSTATICTEXTSTATSCOVERED, 
 wxID_FRAMEQSPACEVIEWSTATICTEXTSTATSREDUNDANT, 
] = [wx.NewId() for _init_ctrls in range(13)]

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class FrameQspaceView(wx.Frame):

    def _init_coll_boxSizerStats_Items(self, parent):
        #The stats panel goes here, before the rest
        parent.AddWindow(self.staticLineSpacer, 0, border=0, flag=wx.EXPAND)
        parent.AddWindow(self.buttonAdvancedView, 0, border=1, flag=0)

    def _init_coll_boxSizerBottomPanel_Items(self, parent):
        parent.AddWindow(self.notebookView, 10, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.panelStats, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)

    def _init_coll_boxSizer1_Items(self, parent):
        parent.AddWindow(self.splitterAll, 1, border=0, flag=wx.EXPAND)

    def _init_sizers(self):
        self.boxSizer1 = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerStats = wx.BoxSizer(orient=wx.VERTICAL)
        self.boxSizerBottomPanel = wx.BoxSizer(orient=wx.HORIZONTAL)

        self._init_coll_boxSizer1_Items(self.boxSizer1)
        self._init_coll_boxSizerStats_Items(self.boxSizerStats)
        self._init_coll_boxSizerBottomPanel_Items(self.boxSizerBottomPanel)

        self.SetSizer(self.boxSizer1)
        self.panelBottom.SetSizer(self.boxSizerBottomPanel)
        self.panelStats.SetSizer(self.boxSizerStats)


    def _init_ctrls(self, prnt):
        wx.Frame.__init__(self, id=wxID_FRAMEQSPACEVIEW,
              name=u'FrameQspaceView', parent=prnt, pos=wx.Point(398, 487),
              size=wx.Size(800, 700), style=wx.DEFAULT_FRAME_STYLE,
              title=u'Reciprocal Space 3D Viewer')
        self.SetClientSize(wx.Size(800, 700))
        self.SetAutoLayout(True)
        self.Bind(wx.EVT_CLOSE, self.OnFrameQspaceViewClose)

        self.splitterAll = wx.SplitterWindow(id=wxID_FRAMEQSPACEVIEWSPLITTERALL,
              name=u'splitterAll', parent=self, pos=wx.Point(0, 0),
              size=wx.Size(778, 792), style=wx.SP_3D)
        self.splitterAll.SetSashSize(8)
        self.splitterAll.SetSashGravity(1.0)

        self.panelBottom = wx.Panel(id=wxID_FRAMEQSPACEVIEWPANELBOTTOM,
              name=u'panelBottom', parent=self.splitterAll, pos=wx.Point(0,
              608), size=wx.Size(778, 184), style=wx.TAB_TRAVERSAL)
        self.panelBottom.SetBackgroundStyle(wx.BG_STYLE_SYSTEM)
        self.panelBottom.SetAutoLayout(True)

        self.panelStats = wx.Panel(id=wxID_FRAMEQSPACEVIEWPANELSTATS,
              name=u'panelStats', parent=self.panelBottom, pos=wx.Point(550, 0),
              size=wx.Size(220, 184), style=wx.TAB_TRAVERSAL)
        self.panelStats.SetBackgroundStyle(wx.BG_STYLE_SYSTEM)
        self.panelStats.SetAutoLayout(True)
        self.panelStats.SetMinSize((220, 168))

        self.panel3D = wx.Panel(id=wxID_FRAMEQSPACEVIEWPANEL3D, name=u'panel3D',
              parent=self.splitterAll, pos=wx.Point(0, 0), size=wx.Size(778,
              600), style=wx.TAB_TRAVERSAL)
        self.panel3D.SetBackgroundColour(wx.Colour(246, 243, 245))
        self.panel3D.SetMinSize(wx.Size(-1, -1))
        self.splitterAll.SplitHorizontally(self.panel3D, self.panelBottom, 520)
        self.panel3D.Bind(wx.EVT_SIZE, self.onPanel3DSize)
        
        self.buttonAdvancedView = wx.Button(id=wxID_FRAMEQSPACEVIEWBUTTONADVANCEDVIEW,
              label=u'3D Advanced Settings...', name=u'buttonAdvancedView',
              parent=self.panelStats, pos=wx.Point(0, 160), size=wx.Size(220,
              24), style=0)
        self.buttonAdvancedView.Bind(wx.EVT_BUTTON,
              self.OnButtonAdvancedViewButton,
              id=wxID_FRAMEQSPACEVIEWBUTTONADVANCEDVIEW)

        self.staticLineSpacer = wx.StaticLine(id=wxID_FRAMEQSPACEVIEWSTATICLINESPACER,
              name=u'staticLineSpacer', parent=self.panelStats, pos=wx.Point(0,
              124), size=wx.Size(220, 36), style=0)

        self.staticTextMouseInfo = wx.StaticText(label=u'Mouse is over:', name=u'staticTextMouseInfo',
              parent=self.panel3D, pos=wx.Point(0, 0), size=wx.Size(125, 17), style=0)

        self.notebookView = wx.Notebook(id=wx.ID_ANY, name=u'notebookView',
            parent=self.panelBottom, pos=wx.Point(0, 0), size=wx.Size(573, 629), style=0)
        self.notebookView.SetMinSize(wx.Size(-1, -1))
        self.notebookView.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.onNotebookPageChanged, id=wx.ID_ANY)

        self._init_sizers()

    def __init__(self, parent):
        self._init_ctrls(parent)

        #Create the qspace options panel
        self.tabVolume = panel_qspace_options.QspaceOptionsPanel(parent=self.notebookView, id=wx.NewId(),
              name=u'tabVolume', pos=wx.Point(40, 16),
              size=wx.Size(624, 120), style=wx.TAB_TRAVERSAL)
        #Add it to the notebook
        self.notebookView.AddPage(self.tabVolume, 'Volume Coverage View', select=True)

        #Create the reflections view options panel
        self.tabReflections = panel_reflections_view_options.PanelReflectionsViewOptions(parent=self.notebookView, id=wx.NewId(),
              name=u'tabReflections', pos=wx.Point(40, 16),
              size=wx.Size(624, 120), style=wx.TAB_TRAVERSAL)
        #Add it to the notebook
        self.notebookView.AddPage(self.tabReflections, 'Reflections View', select=False)

        #Make the stats panel
        self.stats_panel = panel_coverage_stats.PanelCoverageStats(parent=self.panelStats,
            id=wx.NewId(), name=u'stats_panel', pos=wx.Point(0, 0),
            size=wx.Size(624, 120), style=wx.TAB_TRAVERSAL)
        #Put it in the sizer
        self.boxSizerStats.Insert(0, self.stats_panel, 1, border=0, flag=wx.EXPAND)
        
        # Create the view controller
        self.controller = QspaceViewController(self)
        
        #This sets the parent control
        self.control = self.controller.edit_traits(
                        parent=self.panel3D,
                        kind='panel').control
                        
        #Add a sizer that holds the mayaview, and expand it
        sizer = wx.BoxSizer(orient=wx.VERTICAL)
        sizer.Add(self.control, 1, wx.EXPAND)
        #Also put the mouse info line after
        sizer.Add(self.staticTextMouseInfo, 0, border=8, flag=wx.EXPAND | wx.LEFT)
        self.panel3D.SetSizer(sizer)
        
        self.init_plot()

        #Once the window is shown, we can initialize the pickers.
        #  The scene.interactor does not exist until the window is shown!
        self.controller.init_pickers()


    def init_plot(self):
        pass #TODO

    def OnFrameQspaceViewClose(self, event):
        #So that the singleton gets re-created if the window is re-opened
        global _instance
        if self is _instance:
            _instance = None
        #Also clean up the mayaview
        self.controller.cleanup()
        event.Skip()

    def OnButtonAdvancedViewButton(self, event):
#        self.controller.iso.default_traits_view()
        self.controller.show_pipeline()
        event.Skip()

    def onNotebookPageChanged(self, event):
        #Call an update to redraw whatever is needed.
        if hasattr(self, 'controller'):
            wx.CallAfter(self.controller.view_mode_changed)
        event.Skip()

    def onPanel3DSize(self, event):
        #The 3d view is resized
        if not self.controller.in_volume_mode():
            param = display_thread.get_reflection_display_params()
            #Only redraw for automatic pixel size
            if param.automatic_size and param.display_as == model.experiment.ParamReflectionDisplay.DISPLAY_AS_PIXELS:
                self.controller.auto_adjust_pixel_size()
        event.Skip()

if __name__ == '__main__':
    #Test routine
    model.instrument.inst = model.instrument.Instrument()
    model.experiment.exp = model.experiment.Experiment(model.instrument.inst)
    import wx
    class MyApp(wx.App):
        def OnInit(self):
            frame = FrameQspaceView(None)
            frame.Show(True)
            frame.SetClientSize( wx.Size(600, 600) )
            return True
    app = MyApp(0)
    app.MainLoop()
