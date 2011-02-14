"""Module to calculate optimized coverage.
Not designed to run as part of the GUI.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
from random import random
from math import sqrt
import types
import numpy as np

#--- Model Imports ---
import instrument
import experiment

#--- Pygene Imports ---
import pygene
from pygene.gene import FloatGene, FloatGeneMax, FloatGeneRandom
from pygene.organism import Organism, MendelOrganism
from pygene.population import Population

#--- Traits Imports ---
from enthought.traits.api import HasTraits,Int,Float,Str,Property,Bool, List
from enthought.traits.ui.api import View,Item,Label,Heading, Spring, Handler
from enthought.traits.ui.menu import OKButton, CancelButton


# ===========================================================================================
class DetectorOptimizationParameters(HasTraits):
    """Class holding the parameters to perform a genetic algorithm optimization."""
    #list of which detector #s (1-based!) to keep for sure
    keep_list_str = Str('')

    #List of which detectors to exclude from the list for sure (also 1-based)
    exclude_list_str = Str('3,4,5, 12,13,14,15, 21,22,23,24,29,30,31,32,33,40,41,42,43')

    #Number of other detectors to keep (not counting the ones in the keep_list).
    number_to_keep = Int(14)

    popInitSize = Int(20)
    popChildCull = Int(30) # Cull down to this many children after each generation
    popChildCount = Int(100) # Number of children to create after each generation
    popIncest = Int(15)           # Number of best parents to add to children
    popNumMutants = Float(0.7)     # Proportion of mutants each generation
    popNumRandomOrganisms = Int(2)  # Number of random organisms to add per generation
    mutateOneOnly = Bool(False)
    mutateAfterMating = Bool(True) #Check this to mutate all progeny
    crossoverRate = Float(0.15)  #Proportion of genes to split out to first child in each pair resulting from a mating
    geneMutProb = Float(0.6)
    geneMutAmt = Float(0.1)         # only if not using FloatGeneRandom

    header_text = Spring(label="Enter detector ID numbers:", emphasized=True, show_label=True)

    pop_text = Spring(label="Enter population parameters:", emphasized=True, show_label=True)

    gene_text = Spring(label="Enter mutation parameters:", emphasized=True, show_label=True)

    view = View( header_text,
            Item("number_to_keep", label="How many detectors to keep (total)?", format_str="%d", tooltip="Number of other detectors to keep (not counting the ones in the keep_list)."),
            Item("keep_list_str", label="List of detectors to keep for sure:",  tooltip="These detectors will always be included in the list."),
            Item("exclude_list_str", label="List of detectors to exclude:",  tooltip="These detectors will never be included in the list."),
            pop_text, 
            Item("popInitSize", label="Population's initial size:",  tooltip=""),
            Item("popChildCull", label="Children cull down to:",  tooltip="Cull down to this many children after each generation"),
            Item("popChildCount", label="Children count:",  tooltip="Number of children to create after each generation"),
            Item("popIncest", label="Num. of best parents to add to children:",  tooltip="Number of best parents to add to children (incest)."),
            Item("popNumMutants", label="Proportion of mutants:",  tooltip="Proportion of mutants each generation"),
            Item("popNumRandomOrganisms", label="",  tooltip="Number of random organisms to add per generation"),
            gene_text,
            Item("mutateOneOnly", label="Mutate only a single gene?",  tooltip="Dictates whether mutation affects one randomly chosen gene unconditionally, or all genes subject to the genes' individual mutation settings"),
            Item("mutateAfterMating", label="",  tooltip="Check this to mutate all progeny"),
            Item("crossoverRate", label="",  tooltip="Proportion of genes to split out to first child in each pair resulting from a mating"),
            Item("geneMutProb", label="Mutation probability (out of 1):",  tooltip=""),
            Item("geneMutAmt", label="Amount of mutation (out of 1):",  tooltip=""),
            Item("", label="",  tooltip=""),
            title="Enter parameters for genetic algorithm search.",
            buttons=[OKButton, CancelButton],
            width=600, kind='modal'
            )


    def put_in_globals(self):
        """Dump all the traits into the globals namespace."""
        for (name) in self.class_editable_traits():
            globals()[name] = getattr(self, name)


#The global DetectorOptimizationParameters object.
dop = DetectorOptimizationParameters()

#A dictionary where the key is a string of the list of detectors, and the value is the % covered.
saved_fitness = dict()

#Counts of calculations.
count_fitness = 0
count_calc_fitness = 0

#Global genome
genome = dict()


#==========================================================================================
def make_genome(exclude_list, keep_list):
    """ Make a genome. This a dictionary that I think is shared by all organisms.

    exclude_list: list of detector numbers to exclude.
    keep_list: list of detectors that will be kept for sure.
    """
    global genome
    genome.clear()
    #Create 48 float genes
    for i in range(48):
        detnum = i+1
        if not (detnum in keep_list) and not (detnum in exclude_list):
            #This detector is under consideration
            #Gene name MUST BE STRING.
            #Value MUST be the CLASS, not an instance!!!
            genome["%02d" % detnum] = FloatGene
    return genome


#==========================================================================================
#==========================================================================================
class DetectorChoiceGene(FloatGene):
    """Simple float gene giving the priority to a given detector choice."""

    #----------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        global dop
        #Set values
        self.randMin = 0.0
        self.randMax = 1.0
        self.mutProb = dop.geneMutProb
        self.mutAmt = dop.geneMutAmt
        #Call parent constructor.
        super(DetectorChoiceGene, self).__init__(*args, **kwargs)




#==========================================================================================
#==========================================================================================
class DetectorChoice(Organism):
    """An organism which represents a choice of detector coverage."""
    #----------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        global dop, genome
        #Set values
        self.genome = genome
        self.mutateOneOnly = dop.mutateOneOnly
        self.crossoverRate = dop.crossoverRate
        self.numMutants = 0.3
        #Call parent constructor. It builds up the initial population and stuff.
        super(DetectorChoice, self).__init__(*args, **kwargs)

    #----------------------------------------------------------------
    def get_detector_list(self):
        """Return the list of detectors to used, based on the genes of this organism."""
        global dop
        #List of the gene values
        sorter = [(self[detnum], detnum) for (detnum, gene) in self.genome.items()]
        #Sort using the values (first thing in the tuple)
        sorter.sort()
        #List of detectors numbers to use, sorted in order.
        det_list = [val[1] for val in sorter[:dop.number_to_keep]]
        #Add the ones we always keep (converted to strings too)
        det_list = det_list + ["%02d" % val for val in dop.keep_list]
        det_list.sort()
        #Here's the list
        return det_list

    #----------------------------------------------------------------
    def fitness(self):
        """Fitness of this individual"""
        global count_fitness, saved_fitness
        count_fitness += 1
        #Get the list
        det_list = self.get_detector_list()
        #Make it into a keystring
        keystring = ""
        for val in det_list: keystring += val
        #Does the key exist?
        if saved_fitness.has_key(keystring):
            #No need to recalculate! We have it
            return saved_fitness[keystring]
        else:
            fit = self.calc_fitness(det_list)
            #Save it using the keystring
            saved_fitness[keystring] = fit
            #And return it
            return fit

    #----------------------------------------------------------------
    def calc_fitness(self, det_list):
        """Calculate the fitness given a list of detector numbers (1-based!)"""
        global count_calc_fitness
        count_calc_fitness += 1

        #Use all positions
        experiment.exp.params[experiment.PARAM_POSITIONS] = None
        #Use specific detectors ---
        #Make a bool array.
        detectors = np.zeros(48, dtype=np.bool)
        for val in det_list:
            detnum = int(val)-1
            detectors[detnum] = True
        #Save as parameter
        experiment.exp.params[experiment.PARAM_DETECTORS] = experiment.ParamDetectors(detectors)
        #Do the calculationg
        experiment.exp.calculate_coverage()
        #Return the NON-covered % as the fitness value
        return 100 - experiment.exp.overall_coverage
        


#==========================================================================================
#==========================================================================================
class DetectorChoicePopulation(Population):
    """The population of organisms. Just set the values saved above."""

    def __init__(self, *args, **kwargs):
        global dop
        #Set values
        self.initPopulation = dop.popInitSize
        self.species = DetectorChoice
        self.childCull = dop.popChildCull
        self.childCount = dop.popChildCount
        self.incest = dop.popIncest
        self.mutants = dop.popNumMutants
        self.numNewOrganisms = dop.popNumRandomOrganisms
        self.mutateAfterMating = dop.mutateAfterMating
        #Call parent constructor. It builds up the initial population and stuff.
        super(DetectorChoicePopulation, self).__init__(*args, **kwargs)






#==========================================================================================
#==========================================================================================
def optimize_detector_choice(optim_params, gui=False):
    """Run the genetic algorithm that optimizes the choice of detectors.
    
        dop: DetectorOptimizationParameters object.

    Returns:
        det: list of detector numbers that were chosen as the best solution."""
        
    global dop
    print "Starting optimize_detector_choice"
    dop = optim_params

    def list_from_string(in_string):
        """Make a list of ints from a comma-separated list in a string."""
        out = []
        for s in in_string.split(','):
            try:
                out.append( int( float(s) ) )
            except Exception, e:
                pass
        return out

    #Read in the lists
    exclude_list = list_from_string( dop.exclude_list_str )
    keep_list = list_from_string( dop.keep_list_str )
    dop.add_trait('exclude_list',  List(exclude_list))
    dop.add_trait('keep_list', List(keep_list))

    #Genome needs to be initialized before any organisms get made
    make_genome(exclude_list, keep_list)

    #Reduce verbosity
    experiment.exp.verbose = False
    experiment.exp.inst.verbose = False

    # create initial population
    pop = DetectorChoicePopulation()

    #Steps in calculation
    max = 100

    if gui:
        import wx
        prog_dlg = wx.ProgressDialog( "Optimizing detector choice using Genetic Algorithm", "Starting...\n\n\n",
                     max, style = wx.PD_CAN_ABORT | wx.PD_APP_MODAL |  wx.PD_AUTO_HIDE,
                     )
        prog_dlg.SetSize(wx.Size(500, 200))
        prog_dlg.Update(0)

    latest_avg = []
    num_to_quit = 20
    keep_going = True
    skipit = False
    count = 0
    
    while keep_going:
        thebest = pop.best().fitness()
        theavg = pop.fitness()

        #Make a list of the latest values and stop when you don't improve in num_to_quit gens
        latest_avg.append(theavg)
        if len(latest_avg) > num_to_quit and np.allclose(thebest, np.array(latest_avg[-num_to_quit:])): break

        #Build up the best detector list
        det_list = [int(val) for val in pop.best().get_detector_list()]
        message = "Gen. %s: Best=%.2f%%, avg=%.2f%% coverage.\nBest is: %s" % (count, 100-thebest, 100-theavg, det_list)
        print message
        message += "\n\nPress cancel to stop search here."

        count += 1
        if gui:
            out = prog_dlg.Update(100-thebest, message)
            #Return value of this can be tuple or bool, depends on the particular wx version ? or something.
            if type(out)==types.TupleType:
                keep_going = out[0]
            else:
                keep_going = out
            if not keep_going: break

        try:
            #Do a generation
            pop.gen()
        except KeyboardInterrupt:
            keep_going = False
            
    #(end of while loop)

    if gui:
        #Clean up dialog
        prog_dlg.Destroy()

    # get the best solution
    solution = pop.best()

    print "\n\n---------------------------------------\n\n"
    print "The best organism found had coverage of %s%%!" % (100-solution.fitness())
    det = [int(val) for val in solution.get_detector_list()]
    print "The list of detectors it had is %s" % det

    print "\nThe saved_fitness dictionary has %s items." % len(saved_fitness.items())
    print "Fitness called", count_fitness, "times. calc_fitness() called", count_calc_fitness, "times."

    #Restore verbosity
    experiment.exp.verbose = True
    experiment.exp.inst.verbose = True

    return det



#==========================================================================================
def test_setup():
    """Setup a small test."""
    #Create the space
    instrument.inst = instrument.Instrument("TOPAZ_geom_all_2011.csv")
    experiment.exp = experiment.Experiment(instrument.inst)
    experiment.exp.inst.make_qspace()
    experiment.exp.verbose = False
    experiment.exp.inst.verbose = False


    #Angles to search
    chi_list = [0]
    phi_list = np.arange(0, 180, 90)
    omega_list = [0]
    
    #Simulate some positions
    for chi in np.deg2rad(chi_list):
        for phi in np.deg2rad(phi_list):
            for omega in np.deg2rad(omega_list):
                instrument.inst.simulate_position([phi, chi, omega])



#==========================================================================================
#==========================================================================================
#==========================================================================================
if __name__ == "__main__":
    test_setup()
    
    if dop.configure_traits():
        #User clicked okay
        optimize_detector_choice(dop)
    
