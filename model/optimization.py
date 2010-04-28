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
    inst = instrument.inst

    positions = get_angles(chromosome)
    
    #@type exp Experiment
    exp = experiment.exp
    #Calculate
    exp.recalculate_reflections(positions, calculation_callback=None)

    #Return the measured fraction
    score = exp.reflection_stats_with_symmetry.measured * 1.0 / exp.reflection_stats_with_symmetry.total
    score = exp.reflection_stats.measured * 1.0 / exp.reflection_stats.total

#    print  "Fitness score was %s" % score

    return score


#-----------------------------------------------------------------------------------------------
def run_main(num_positions):
    #The instrument to use
    inst = instrument.inst
    exp = experiment.exp
    exp.verbose = False

    # Genome instance, 1D List of 50 elements
    num_angles = len(inst.angles)
    genome = G1DList.G1DList( num_angles * num_positions )

    #Make the initializator a real value
    genome.initializator.set(Initializators.G1DListInitializatorReal)

    # Sets the range max and min of the 1D List
    genome.setParams(rangemin=-np.pi, rangemax=np.pi)

    # The evaluator function (evaluation function)
    genome.evaluator.set(eval_func)

    # Genetic Algorithm Instance
    ga = GSimpleGA.GSimpleGA(genome)
    ga.setMultiProcessing(True, full_copy=False)
    ga.setPopulationSize(100)

    # Set the Roulette Wheel selector method, the number of generations and
    # the termination criteria
    ga.selector.set(Selectors.GRouletteWheel)
    ga.setGenerations(100)
    ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)

    # Sets the DB Adapter, the resetDB flag will make the Adapter recreate
    # the database and erase all data every run, you should use this flag
    # just in the first time, after the pyevolve.db was created, you can
    # omit it.
    sqlite_adapter = DBAdapters.DBSQLite(identify="ex1", resetDB=True)
    ga.setDBAdapter(sqlite_adapter)

    # Do the evolution, with stats dump
    # frequency of 20 generations
    ga.evolve(freq_stats=1)

    # Best individual
    best = ga.bestIndividual()

    exp.verbose = True

    #Return the positions
    return get_angles(best)


if __name__ == "__main__":
    #Inits
    instrument.inst = instrument.Instrument("../instruments/TOPAZ_detectors_2010.csv")
    experiment.exp = experiment.Experiment(instrument.inst)
    exp = experiment.exp
    exp.initialize_reflections()
    exp.verbose = False
    #Run
    print run_main(10)

