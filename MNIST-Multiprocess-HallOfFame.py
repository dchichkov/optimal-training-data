''' 
Insight.  Data is the new source code.  Applying evolutionary or other 
optimization algoritms to 'optimize' that source code is interesting.
Goals of optimization - increase accuracy, training spead, etc.

Low hanging fruits are:
 * optimal subsampling of datasets;
 * optimizing datasets for single-epoch training;
 * optimizing datasets for finetuning;

Goal: Subsample the MNIST training set to provide minimal and optimal set of examples 
that maximizes a test score of an arbitrary neural network trained on that set.

Example: Code below evolves indexes (a subset of 4096 samples from MNIST dataset) 
to optimize validation accuracy of a simple convnet trained for 1 epochs on that subset.

It trains another convnet (one extra layer of convolutions) on the same indexes 
and evaluates its accuracy on the test set, to get the *test accuracy*.

If the 'optimal' dataset is somewhat general between architecures, we should
see the increase of test accuracy during evolution of the indexes.
'''

from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random, numpy, pylab as pl, collections, multiprocessing, time, json, gc
# GA population size. Note, set to small value (5) if you'd like to see 
# algorithm do. GA would first evaluate fitness of the population, which 
# takes population_size train/valid runs of the MNIST. GThe default is 200.
population_size = 200

# GA hall_of_fame_size. 
hall_of_fame_size = 20

# The size of the dataset to optimize
dataset_size = 4096
len_x_train = 30000   # should be equal to len(x_train)

def f(indexes, is_test = False):
    """
      input: indexes to train the model on
      is_test:
        false: returns 'training' model accuracy on the validation set
        true: returns 'test' model accuracy on the test set
    """
    gc.collect()
    import MNIST
    return MNIST.f(indexes, is_test)


#print("Initial accuracy:", f(range(dataset_size)))
#print("Initial Test accuracy:", f(range(dataset_size), is_test = True))
# initial = model.get_weights()   # initial trained parameters
  
# evolve the dataset to reach better accuracy on the validation set
from deap import algorithms, base, creator, tools
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

pool = multiprocessing.Pool(processes = 4)
toolbox.register("map", pool.map)

# each gene is dataset_size of example indexes
toolbox.register("indices", random.sample, range(len_x_train), dataset_size)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", f)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

random.seed(64)  
pop = toolbox.population(n=population_size)


stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

class HallOfFame(tools.HallOfFame):
    def update(self, population):
        super().update(population)

        for individual in self:
            if not hasattr(individual, 'test_accuracy'):
                individual.test_accuracy = f(individual, is_test = True)
                print("Adding to Hall Of Fame, Fitness:", individual.fitness.values, "Test Accuracy:", individual.test_accuracy)
                with open("HallOfFame.json", "a+") as json_file:
                    json.dump({'individual' : individual, 
                               'fitness' : individual.fitness.values, 
                               'test_accuracy' : individual.test_accuracy}, 
                              json_file)
                    json_file.write("\n")

hof = HallOfFame(hall_of_fame_size)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)
    
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=2000, 
                               stats=stats, halloffame=hof, verbose=True)
    

