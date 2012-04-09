""" mlab utils module

Provides useful functions that extend the mlab (enthought.mayavi.mlab)
3d plot scripting tools.
"""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import numpy as np

#--- GUI Imports ---
import enthought.mayavi.mlab as mlab_module

#--- Model Imports ---
from model.numpy_utils import *


#------------------------------------------------------------
def lines(points,  plotobj=None, color=(0,0,0), head_size=0, mlab=mlab_module, **kwargs):
    #Turn into arrays
    x=np.array([pt[0] for pt in points])
    y=np.array([pt[1] for pt in points])
    z=np.array([pt[2] for pt in points])

    if plotobj is None:
        #Create a new plot
        radius = 2
        #Tube radius
        tube_radius = kwargs.get("tube_radius", None)
        if tube_radius is None:
            rep = 'wireframe'
        else:
            rep = 'surface'

        return mlab.plot3d(x,y,z, line_width=radius, representation=rep, color=color, **kwargs)
    else:
        #Change the data in it
        plotobj.mlab_source.reset(x=x,y=y,z=z, color=color)
        return plotobj
    
#------------------------------------------------------------
def arrow(start, stop, plotobj=None, color=(0,0,0), head_size=0, mlab=mlab_module, **kwargs):
    """Draws an arrow from start to stop (each are 3-element vectors).
    
    Parameters:
        color: e.g. (1, 1, 1) for white.
        head_size: size in space units of the arrow head.
    """
    points = [start, stop]
    if head_size>0:
        #Add the arrow head
        #The vector we are drawing, normalized
        diff = (stop-start)
        diff = vector(diff) / vector_length(diff)
        #Cross vector product of diff and a vertical vector gives the plane of the head.
        head = np.cross(diff, vector([0,1.,0.]))
        #Avoid divide by 0, pick a non-parallel one
        if vector_length(head) < 1e-5:
            head = np.cross(vector(diff), vector([1.,0.,0.]))
        #Normalize
        head = head / vector_length(head)
        #Vectors defining the head
        a = diff * head_size
        b = head * head_size
        points.append( vector(stop) - a + b)
        points.append( vector(stop) )
        points.append( vector(stop) - a - b)

    return lines(points,  plotobj, color, head_size, **kwargs)


#------------------------------------------------------------
def plot3dpatch(vectors, color, alpha, plotobj=None):
    """Plots a matrix containing vertical XYZ vectors for each point to form a closed patch.
    Currently only does a single triangle."""
    x = vectors[0, :]
    y = vectors[1, :]
    z = vectors[2, :]
    triangles = [(0,1,2)]
    if plotobj is None:
        #Create a new plot
        return mlab.triangular_mesh(x,y,z,triangles, color=color, opacity=alpha)
    else:
        #Change the data in it
        plotobj.mlab_source.set(x = x, y=y, z=z)
        return plotobj
    
#------------------------------------------------------------
def plot3dpoints(vectors, scale_factor=10, plotobj=None, mlab=mlab_module, **kwargs):
    """Plots a matrix containing vertical XYZ vectors for each point."""
    if (vectors.ndim == 1):
        vectors = vectors.reshape(3, 1); #Force into column
    if plotobj is None:
        #Create a new plot
        return mlab.points3d(vectors[0,:], vectors[1,:], vectors[2,:], scale_factor=scale_factor, **kwargs)
    else:
        #Change the data in it
        plotobj.mlab_source.set(x = vectors[0,:], y=vectors[1,:], z=vectors[2,:])
        return plotobj
    
#------------------------------------------------------------
def text3d(vector, text, font_size=20, vertical_justification="center", horizontal_justification="center", color=(1.,1.,1.), mlab=mlab_module, **kwargs):
    """Place text in 3d using a vector as input.
    Parameters:
        vector: 3-vector with the x,y,z coords.
        text: string to show
        font_size: point size of font. If it is a float < 1, relative scaling is used instead. The
            size is the fraction of the window size.
        horizontal_justification and horizontal_justification: alignment of text.
        color: RGB tuple of color
    """
    if font_size<=1:
        #Scaled
        txt = mlab.text(vector[0], vector[1], z=vector[2], text=text, width=font_size, color=color, **kwargs)
    else:
        #Fixed font
        txt = mlab.text(vector[0], vector[1], z=vector[2], text=text, width=0.1, color=color, **kwargs)
        txt.actor.text_scale_mode = "none"
        txt.actor.text_property.font_size = font_size
        
    txt.actor.text_property.vertical_justification = vertical_justification
    txt.actor.text_property.justification = horizontal_justification
        


#------------------------------------------------------------
def draw_cartesian_axes(size=10, offset=np.array([0,0,0]), textwidth=0.006):
    """Draws the x/y/z axes in the current figure, using the ARROW function.
       size: size to draw the arrows.
       offset: vector to offset the center of the axes with"""
    zero = np.array([0, 0, 0]) + offset
    arrow(zero, np.array([1, 0, 0]) * size + offset, color=(1,0,0));
    arrow(zero, np.array([0, 1, 0]) * size + offset, color=(0, 0.5, 0))
    arrow(zero, np.array([0, 0, 1]) * size + offset, color=(0,0,1))
    
    mlab.text(size + offset[0], offset[1], z=offset[2], text='X', width=textwidth)
    mlab.text(offset[0], size + offset[1], z=offset[2], text='Y', width=textwidth)
    mlab.text(offset[0], offset[1], z=(size + offset[2]), text='Z', width=textwidth)


