import random, copy, numpy as np
import train_rbfn

from itertools import repeat, combinations
from cromossome import Cromossome
from math import log

class GeneticAlgorithm:
    def __init__(self, training, test):
        self.training = training
        self.test = test
        self.pop = []
        self.min_size = 2
        self.max_size = training["INPUT"].size - 1

        self.training_input  = np.asarray([[x.tolist()] for x in training["INPUT"].values])
        self.training_output = np.asarray([[x.tolist()] for x in training["OUTPUT"].values])

    def init_pop(self, size):
        for _ in repeat(None, size):
            individual = Cromossome()
            individual.string = random.sample(range(0, len(self.training["INPUT"])), random.randint(self.min_size, self.max_size))
            self.pop.append(individual)

    def decode(self, cromossome):
        centers = []
        for center in cromossome:
            centers.append([self.training["INPUT"].values[center].tolist()])

        return np.asarray(centers)

    def evalue(self, cromossome):
        net = train_rbfn.train(self.training_input, self.training_output, self.decode(cromossome.string))
        cromossome.fitness = [net.aic(self.training), net.aic(self.test)]

    def evolve(self, generations):
        generation = 1
        for _ in repeat(None, generations):
            print("Generation ", generation)
            generation += 1

            for individual in self.pop:
                self.evalue(individual)
            self.rank_all()

            new_population = []
            while len(new_population) < len(self.pop):
                parents = [self.select_individual() for _ in range(2)]
                offsprings = self.crossover(parents) if random.random() <= 0.90 else [copy.deepcopy(i) for i in parents]

                for offspring in offsprings:
                    self.mutation(offspring)
                    self.deletion(offspring)
                    self.addition(offspring)
                    if offspring.string != parents[0].string and offspring.string != parents[1].string:
                        self.evalue(offspring)
                        parents.append(offspring)

                # Compara a aptidão dos descendentes e os pais. Os dois melhores são selecionados
                selected = []
                while len(selected) < 2:
                    for individual in parents:
                        individual.p_optimal = True

                    self.compare(parents)
                    selected.extend([individual for individual in parents if individual.p_optimal][:2 - len(selected)])
                    parents = [e for e in parents if not e.p_optimal]

                new_population.extend(selected)

            self.pop = new_population

        for individual in self.pop:
                individual.p_optimal = True

        self.compare(self.pop)
        best_config = [individual for individual in self.pop if individual.p_optimal][0]
        print(self.decode(best_config.string))

        return train_rbfn.train(self.training_input, self.training_output, self.decode(best_config.string))

    def compare(self, population):
        for a, b in combinations(population, 2):
            if a.partially_less_than(b):
                b.p_optimal = False
            elif b.partially_less_than(a):
                a.p_optimal = False

    def rank_all(self):
        new_pop = []
        current_rank = 0

        while self.pop:
            for individual in self.pop:
                individual.p_optimal = True

            self.compare(self.pop)

            for individual in self.pop:
                if individual.p_optimal:
                    individual.rank = len(self.pop) - current_rank
                    new_pop.append(individual)

            self.pop = [individual for individual in self.pop if not individual.p_optimal]

            current_rank+=1

        self.pop = new_pop

    def select_individual(self):
        max  = sum(individual.rank for individual in self.pop)
        pick = random.uniform(0, max)
        current = 0
        for individual in self.pop:
            current += individual.rank
            if current > pick:
                return individual

    def crossover(self, parents):
        if len(parents[0].string) <= len(parents[-1].string):
            smaller, bigger = [copy.deepcopy(parent) for parent in parents]
        else:
            bigger, smaller = [copy.deepcopy(parent) for parent in parents]

        common = set(smaller.string).intersection(bigger.string)
        diff   = list(set(smaller.string) - common)

        # Selecionando genes para troca
        if diff:
            quantity = random.randint(1, len(diff))
            first_sample = [diff[i] for i in sorted(random.sample(range(len(diff)), quantity))]
            diff = list(set(bigger.string) - common)

            if diff:
                second_sample = [diff[i] for i in sorted(random.sample(range(len(diff)), quantity))]

                findex = [smaller.string.index(item) for item in first_sample]
                sindex = [bigger.string.index(item) for item in second_sample]

                for i, j in zip(findex, sindex):
                    smaller.string[i], bigger.string[j] = bigger.string[j], smaller.string[i]

        return [smaller, bigger]

    def mutation(self, cromossome):
        for locus in range(len(cromossome.string)):
            if random.random() <= 0.01:
                cromossome.string[locus] = random.sample(set(range(0, len(self.training["INPUT"]))) - set(cromossome.string), 1)[0]

    def deletion(self, cromossome):
        if random.random() <= 0.01:
            starting_from = random.randint(0, len(cromossome.string) - 1)
            upto = starting_from + random.randint(1, len(cromossome.string) - starting_from)
            if (upto - starting_from) == len(cromossome.string):
                upto -= 1
            cromossome.string = [gene for index, gene in enumerate(cromossome.string) if index not in range(starting_from, upto)]

    def addition(self, cromossome):
        if random.random() <= 0.01 and len(cromossome.string) < self.max_size:
            quantity = random.randint(1, self.max_size - len(cromossome.string))
            cromossome.string.extend(random.sample(set(range(0, len(self.training["INPUT"]))) - set(cromossome.string), quantity))
