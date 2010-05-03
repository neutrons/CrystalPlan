""" Optimization of various parameters
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id: goniometer.py 1174 2010-04-23 14:58:40Z 8oz $

#--- General Imports ---
import instrument
import experiment

import pyevolve
print "pyevolve is version", pyevolve.__version__
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Statistics
from pyevolve import DBAdapters
from pyevolve import Initializators

import numpy as np


#--- Traits Imports ---
from enthought.traits.api import HasTraits,Int,Float,Str,Property,Bool, List
from enthought.traits.ui.api import View,Item,Label,Heading, Spring, Handler, Group
from enthought.traits.ui.menu import OKButton, CancelButton


# ===========================================================================================
class OptimizationParameters(HasTraits):
    number_of_orientations = Int(10, desc="the number of orientations you want in the sample plan.")
    desired_coverage = Float(85.0, desc="the percent reciprocal-space coverage you want. The optimization will stop when it reaches this point.")
    use_symmetry = Bool(False, label='Use crystal symmetry', desc="to consider crystal symmetry in determining reflection coverage.")
    auto_increment = Bool(True, label='Auto increment # of orientations?', desc="that if the optimization does not converge in the # of generations, add one to the # of sample orientations and try again.")

    population = Int(100, desc="the number of individuals to evolve.")
    max_generations = Int(100, desc="the maximum number of generations to evolve before giving up.")
    mutation_rate = Float(0.05, desc="the probability of mutation per individual.")
    use_multiprocessing = Bool(True, desc="to use multiprocessing (multiple processors) to speed up calculation.")

    view = View(
        Group(
            Item('number_of_orientations'),
            Item('desired_coverage'),
            Item('use_symmetry'),
            Item('auto_increment'),
            label='Optimization Settings'
        ),
        Group(
            Item('population'),
            Item('max_generations'),
            Item('mutation_rate'),
            Item('use_multiprocessing'),
            label='Genetic Algorithm Settings'
        ),
        Spring(label=' ')
        )
    


#-----------------------------------------------------------------------------------------------
def get_angles(chromosome):
    """Extract the list of lists of angles"""
    inst = instrument.inst
    exp = experiment.exp
    
    num_angles = len(inst.angles)
    num_positions = len(chromosome)/num_angles
    umatrix = exp.crystal.get_u_matrix()

    #Create a list of positions
    positions = []
    for i in xrange(num_positions):
        angles = chromosome[i*num_angles:(i+1)*num_angles]
        positions.append(  instrument.PositionCoverage(angles, coverage=None, sample_U_matrix=umatrix) )
        
    return positions

#-----------------------------------------------------------------------------------------------
def eval_func(chromosome):
    """Fitness evaluation function for a chromosome in coverage optimization."""
    global op #@type op OptimizationParameters
    inst = instrument.inst

    positions = get_angles(chromosome)
    
    #@type exp Experiment
    exp = experiment.exp
    #Calculate
    exp.recalculate_reflections(positions, calculation_callback=None)

    #Return the measured fraction
    if op.use_symmetry:
        score = exp.reflection_stats_with_symmetry.measured * 1.0 / exp.reflection_stats_with_symmetry.total
    else:
        score = exp.reflection_stats.measured * 1.0 / exp.reflection_stats.total
#    print  "Fitness score was %s" % score

    return score


#-----------------------------------------------------------------------------------------------
def termination_func(ga_engine):
    """Termination function for G.A. terminates evolution when
    the desired fitness (coverage) is reached."""
    global op #@type op OptimizationParameters
    best_score = ga_engine.bestIndividual().score
    #When you reach the desired coverage (in %) you are done.
    return best_score * 100.0 >= op.desired_coverage
    

#-----------------------------------------------------------------------------------------------
def run_optimization(optim_params, step_callback=None):
    """
    Parameters:
        optim_params: OptimizationParameters object with the parameters
        step_callback: function called after every generation, that
            returns True to abort the optimization.
    """
    global op #@type op OptimizationParameters
    op = optim_params

    #The instrument to use
    inst = instrument.inst
    exp = experiment.exp
    exp.verbose = False

    # Genome instance, 1D List of 50 elements
    num_angles = len(inst.angles)
    genome = G1DList.G1DList( num_angles * op.number_of_orientations )

    #Make the initializator a real value
    genome.initializator.set(Initializators.G1DListInitializatorReal)

    # Sets the range max and min of the 1D List
    genome.setParams(rangemin=-np.pi, rangemax=np.pi)

    # The evaluator function (evaluation function)
    genome.evaluator.set(eval_func)

    # Genetic Algorithm Instance
    ga = GSimpleGA.GSimpleGA(genome)

    #Set the GA parameters from the configuration variable
    ga.setMutationRate(op.mutation_rate)
    ga.setMultiProcessing(op.use_multiprocessing, full_copy=False)
    ga.setPopulationSize(op.population)
    ga.setGenerations(op.max_generations)

    #This is the function that can abort the progress.
    ga.stepCallback.set(step_callback)
    #And this is the termination function
    ga.terminationCriteria.set(termination_func)

    # Set the Roulette Wheel selector method
    ga.selector.set(Selectors.GRouletteWheel)

    # Sets the DB Adapter, the resetDB flag will make the Adapter recreate
    # the database and erase all data every run, you should use this flag
    # just in the first time, after the pyevolve.db was created, you can
    # omit it.
    sqlite_adapter = DBAdapters.DBSQLite(identify="ex1", resetDB=True)
    ga.setDBAdapter(sqlite_adapter)

    # Do the evolution, with stats dump freq
    freq_stats = 0
    if __name__ == "__main__": freq_stats = 1
    results = ga.evolve(freq_stats=freq_stats)

    exp.verbose = True

    return results


if __name__ == "__main__":
    #Inits
    instrument.inst = instrument.Instrument("../instruments/TOPAZ_detectors_2010.csv")
    experiment.exp = experiment.Experiment(instrument.inst)
    exp = experiment.exp
    exp.initialize_reflections()
    exp.verbose = False
    #Run
    op=OptimizationParameters()
    op.desired_coverage = 85
    op.number_of_orientations = 10
    op.mutation_rate = 0.05
    op.use_symmetry = True
    op.max_generations = 1000
    op.use_multiprocessing = True
    print run_main( op )

