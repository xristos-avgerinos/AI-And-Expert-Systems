import random

# a dictionary that contains all the routes and their cost.
route_cost = {
    "AB": 4,
    "AC": 4,
    "AD": 7,
    "AE": 3,
    "BC": 2,
    "BD": 3,
    "BE": 5,
    "CD": 2,
    "CE": 3,
    "DE": 6
}

# we represent the 5 cities with binary system.
# encoding
city_to_binary = {
    "A": "000",
    "B": "001",
    "C": "010",
    "D": "011",
    "E": "100"
}


def create_chromosome(l):
    """
    Create a random chromosome from the list(l) that contains the cities.
    :param l: the list with all the cities.
    :return: A random chromosome.
    """

    return random.shuffle(l)


def get_key_from_value(dic, val):
    """
    Finds the key that corresponds the specific value in the dictionary.
    :param dic: the dictionary we want to search the key from the value.
    :param val: the value we search in the dictionary.
    :return: the key that corresponds the specific value in the dictionary.
    """
    keys = [k for k, v in dic.items() if v == val]
    if keys:
        return keys[0]
    return None


def calculate_fitness(chromosome):
    """
    Calculating the fitness of the given chromosome by adding all routes' cost.
    :param chromosome: list that represents the chromosome we search for fitness.
    :return: the fitness of the given chromosome.
    """
    fitness = 0
    for i in range(len(chromosome) - 1):
        starting_city = get_key_from_value(city_to_binary, chromosome[i])
        terminal_city = get_key_from_value(city_to_binary, chromosome[i + 1])
        route = starting_city + terminal_city
        try:
            fitness += route_cost[route]
        except KeyError:
            # we have exception when the route doesn't exist into the dictionary, and so we have to read it upside
            # down.For example the rout 'BA' doesn't exist into the dictionary, but it has the same cost with 'AB'
            fitness += route_cost[route[::-1]]
    return fitness


def select_parents(population, population_fitness_score, fitness_sum):
    """
    Parent selection process using the Roulette Wheel method.
    :param population: a list that represents the population that contains all the randomly generated chromosomes.
    :param population_fitness_score: a list parallel to population list which contains the fitness of each chromosome into population.
    :param fitness_sum: an integer that represents the summary of all chromosomes' fitness.
    :return: A list with the parents to mate.
    """
    mating_pool = []

    while len(mating_pool) < len(population):
        # generate a random number between (0, fitness_sum)
        # in order to spin the wheel and select a fitness value
        rand = random.uniform(0, fitness_sum)
        # print(rand)

        # recalculate a temporary sum of fitness values. If the
        # temporary sum is >= of the random number, append the
        # chromosome at that position to the mating pool.
        temp_sum = 0
        for index, score in enumerate(population_fitness_score):
            temp_sum += score
            if temp_sum >= rand:
                mating_pool.append(population[index])
                break

    return mating_pool


def breed(parent1, parent2):
    """
    The modified crossover process(breed), which always give valid chromosomes in mating.
    :param parent1: a list which represents the first parent we will use to mate.
    :param parent2: a list which represents the second parent we will use to mate.
    :return: the descendant(offspring) that comes from the two parents.
    """

    # we keep the two intermediate cities of the first parent and completing the rest
    # from the second parent going from right to left
    descendant = parent1.copy()
    descendant.pop(1)
    descendant.pop(3)
    if parent2[4] not in descendant:
        descendant.insert(3, parent2[4])
    elif parent2[1] not in descendant:
        descendant.insert(3, parent2[1])
    elif parent1[4] not in descendant:
        descendant.insert(3, parent1[4])

    for value in city_to_binary.values():
        if value not in descendant:
            descendant.insert(1, value)

    return descendant


def crossover(mating_pool):
    """
    Gathers all the descendants which come from the mating pool into a list.
    :param mating_pool: a list that contains all the parents we will use for mating.
    :return: A new generation(list) of chromosomes.
    """
    new_generation = []

    for i in range(0, len(mating_pool) - 1, 2):
        parent1 = mating_pool[i].copy()
        parent2 = mating_pool[i + 1].copy()

        descendant1 = breed(parent1, parent2)
        descendant2 = breed(parent2, parent1)
        print('-'.join(x for x in parent1), "            ", '-'.join(x for x in parent2), "            ",
              '-'.join(x for x in descendant1), "            ", '-'.join(x for x in descendant2))
        new_generation.append(descendant1)
        new_generation.append(descendant2)

    if len(mating_pool) % 2 == 1:
        new_generation.append(mating_pool[-1])
    return new_generation


def mutatePopulation(population, mutationRate):
    """
    Performs mutation to a rate(percentage) of the population after the production of the new generation(crossover)
    :param population: a list that represents the population that contains all the chromosomes(after crossover).
    :param mutationRate: the mutation rate(percentage)
    :return: the new mutated population
    """
    mutatedPopulation = []

    for ind in range(0, len(population)):

        chromosome = population[ind]
        if random.random() <= mutationRate:

            while True:
                swapped = int(random.random() * len(chromosome))
                swapWith = int(random.random() * len(chromosome))
                if swapped != 0 and swapped != 5 and swapWith != 0 and swapWith != 5 and swapped != swapWith:
                    break
            city1 = chromosome[swapped]
            city2 = chromosome[swapWith]
            chromosome[swapped] = city2
            chromosome[swapWith] = city1
        mutatedChromosome = chromosome
        mutatedPopulation.append(mutatedChromosome)

    return mutatedPopulation


def partial_renewal(population, population_fitness_score, PARTIAL_RENEWAL):
    """
    Gathers the percentage of the population that will remain unchanged(without crossover) and the percentage that
    will be used in crossover with their fitness scores.
    :param population: a list that represents the population that contains all the chromosomes(before crossover).
    :param population_fitness_score: a list parallel to population list which contains the fitness of each chromosome into population.
    :param PARTIAL_RENEWAL: the percentage of population that will be performed crossover .
    :return: a list with the remained population,a list with the population that will be performed crossover
    and the parallel list with the fitness scores.
    """
    remainedPopulation = []

    n = int(len(population) * round(1 - PARTIAL_RENEWAL, 2))

    for i in range(n):
        x = random.choice(population)

        remainedPopulation.append(x)
        index = population.index(x)

        population.remove(x)
        population_fitness_score.pop(index)

    partPopulation = population

    partFitness = population_fitness_score

    return partPopulation, remainedPopulation, partFitness


def main():
    bin_list = list(city_to_binary.values())
    bin_list.remove("000")

    MUTATION_RATE = 0.001
    PARTIAL_RENEWAL = 0.8

    population = []
    population_fitness_score = []
    POPULATION_SIZE = 100
    for i in range(POPULATION_SIZE):
        # creating an initial random population, calculating the fitness scores
        create_chromosome(bin_list)
        temp_list = bin_list.copy()
        temp_list.insert(0, "000")
        temp_list.append("000")
        population.append(temp_list)
        population_fitness_score.append(calculate_fitness(temp_list))

    for iteration in range(1, 201):
        # we set a limit if the convergence to the perfect solution (minimum cost) does not work
        print('\n                                                       %dst iteration' % iteration)
        reverse_population_fitness_score = []  # we reverse the costs because the genetic algorithm works upside down
        k = 1
        print("\n      population                       fitness-cost")
        for i, j in zip(population, population_fitness_score):
            print('P%d' % k, '-'.join(x for x in i), '       1/%d' % j, '=', 1 / j)
            reverse_population_fitness_score.append(1 / j)
            k += 1

        # check the convergence
        unique_values = list(set(population_fitness_score))
        valuesCount = [population_fitness_score.count(value) for value in unique_values]
        percentageOfEachValue = [ele / POPULATION_SIZE * 100 for ele in valuesCount]
        max_percentage = max(percentageOfEachValue)
        # if the convergence is more than 95% so we have a best solution-shortest path and so we terminate the
        # algorithm
        if max_percentage >= 95.0:
            max_perc_index = percentageOfEachValue.index(max_percentage)
            population_fitness_index = population_fitness_score.index(unique_values[max_perc_index])
            least_route_cost = population_fitness_score[population_fitness_index]
            best_route = population[population_fitness_index]
            for i in range(len(best_route)):
                best_route[i] = get_key_from_value(city_to_binary, best_route[i])

            best_route_string = "→".join(best_route)
            print("\nThe population converges more than 95% to the cost: " + str(least_route_cost))
            print("\nSo one of the shortest paths-routes is: " + best_route_string + " with minimum cost: " + str(
                least_route_cost))
            break

        # calculate the sum of fitness values for all the chromosomes
        sum_scores = sum(reverse_population_fitness_score)
        print("\nsum of scores: ", sum_scores)
        population, remainedPopulation, reverse_population_fitness_score = partial_renewal(population,
                                                                                           reverse_population_fitness_score,
                                                                                           PARTIAL_RENEWAL)
        sum_scores = sum(reverse_population_fitness_score)
        mating_pool = select_parents(population, reverse_population_fitness_score, sum_scores)

        # we replace the population with the next generation
        # and recalculate the fitness score for the new population
        print()
        print("Parent selection using the Roulette Wheel method and perform mating to a", PARTIAL_RENEWAL * 100,
              "% percentage of the population in order to produce the next generation \n("
              "population) using modified cross-breeding and mutation to a", MUTATION_RATE * 100,
              "% percentage of the new generation(population) so as to mix more the genes on chromosomes.")
        print()
        print(
            "      parent1                              parent2                              descendant1                          descendant2")
        next_generation = crossover(mating_pool)
        population = remainedPopulation + next_generation
        population = mutatePopulation(population, MUTATION_RATE)
        population_fitness_score = []
        for chromosome in population:
            population_fitness_score.append(calculate_fitness(chromosome))

        if iteration == 200:
            print("The algorithm didnt find the shortest path and the minimun cost")


if __name__ == '__main__':
    main()
