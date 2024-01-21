import random
import numpy as np
import copy

environment = [['A',0, 0, 1], 
                [0, 0, 0, 0], 
                [0, 0, 1, 0], 
                [0, 0, 0,'T']]
environment = [['A',0, 0, 1, 0], 
                [0, 0, 0, 0, 0], 
                [0, 0, 1, 1, 0],
                [0, 0, 1, 'T', 0], 
                [0, 0, 0, 0,1]]
agent_row = 0 # the position of agent is in which row
agent_column = 0 # the position of agent is in which column
target_row = 3
target_column = 3
population_size = 100
population = []
pm = 0.1
number_of_iterations = 100
length_of_individual_initialization = 8
mean, standard_deviation = 0, 1 # for mutation 2


def initialization(population_size):
    # possible_actions = [0, 1, 2, 3]
    for i in range(population_size):
        population.append(np.random.randint(1, 5, size=random.randint(1, length_of_individual_initialization)))
    print(population)
    return population

# def movement(action):


def run(individual, location_of_individuals_at_each_step):
    # TODO: this two lines should be clean
    agent_row = 0 # the position of agent is in which row
    agent_column = 0 # the position of agent is in which column
    
    env = copy.deepcopy(environment)
    location_of_individuals_at_each_step.append([])
    # individual = [1, 1, 1, 4, 1, 4, 4]
    # agent_column = env[agent_row].index('A')
    for action in individual:
        match action:
            case 1:
                if agent_column + 1 < len(env[0]) and env[agent_row][agent_column + 1] != 1:
                    env[agent_row][agent_column] = 0
                    env[agent_row][agent_column + 1] = 'A'
                    agent_column += 1
            case 2:
                if agent_row - 1 >= 0 and env[agent_row - 1][agent_column] != 1:
                    env[agent_row][agent_column] = 0
                    env[agent_row - 1][agent_column] = 'A'
                    agent_row -= 1
            case 3:
                if agent_column - 1 >= 0 and env[agent_row][agent_column - 1] != 1:
                    env[agent_row][agent_column] = 0
                    env[agent_row][agent_column - 1] = 'A'
                    agent_column -= 1
            case 4:
                if agent_row + 1 < len(env) and env[agent_row + 1][agent_column] != 1:
                    env[agent_row][agent_column] = 0
                    env[agent_row + 1][agent_column] = 'A'
                    agent_row += 1
        location_of_individuals_at_each_step[-1].append([agent_row, agent_column])
        if [agent_row, agent_column] == [target_row, target_column]:
            # print(env)
            return env
    
    # print(env)
    return env





def distance_between_agent_and_goal(env):
    for i in range(len(env)):
        for j in range(len(env[0])):
            if env[i][j] == 'A':
                agent_row = i
                agent_column = j
    return abs(target_row - agent_row) + abs(target_column - agent_column)

def fitness(population):
    fitnesses = []
    location_of_individuals_at_each_step = []
    for individual in population:
        # print('----')
        # print(individual)
        env = run(individual, location_of_individuals_at_each_step)
        distance = distance_between_agent_and_goal(env)
        is_target_visited = 0 if env[target_row][target_column] == 'T' else 1
        fitnesses.append(is_target_visited * 10 - distance * 3 - int(0.2 * len(individual)))
        # print(fitnesses)
    return fitnesses, location_of_individuals_at_each_step
        

def test(population, fitnesses):
    print('best solution:')
    print(max(fitnesses))
    print(population[fitnesses.index(max(fitnesses))])
        

# implement ranked-based selection
def ranked_based_selection(fitnesses):
    for i in range(len(fitnesses)):
        fitnesses[i] = fitnesses[i] + 20
    # Given list of fitnesses
    # fitnesses = [2, 4, 3, 7, 13, 2]

    # Rank the fitnesses, highest first
    # The rank is the index in the sorted list
    ranked_fitnesses = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)

    # Assign selection probabilities based on rank
    # Here we assign the probability in direct proportion to the rank
    # The highest rank gets probability len(fitnesses), and the lowest gets 1
    selection_probabilities = [len(fitnesses) - rank for rank in range(len(fitnesses))]

    # Select the index based on the assigned probabilities
    # random.choices returns a list, so we take the first element using [0]
    selected_index = random.choices(ranked_fitnesses, weights=selection_probabilities, k=1)[0]

    # Return the selected index
    return selected_index

def roulette_wheel_selection(fitnesses_):
    fitnesses = copy.deepcopy(fitnesses_)
    for i in range(len(fitnesses)):
        fitnesses[i] = fitnesses[i] + abs(min(fitnesses_))
        fitnesses[i] = int(fitnesses[i] * fitnesses[i]) + 1 # add one to give a chance to all individuals
    # print('++++++++++')
    # print(max(fitnesses))
    # print('++++++++++')
    total_fitness = sum(fitnesses)
    selection_probs = [f/total_fitness for f in fitnesses]
    # print(selection_probs)
    # print(fitnesses)
    # Create a cumulative sum of the selection probabilities
    cumulative_probs = [sum(selection_probs[:i+1]) for i in range(len(selection_probs))]
    # print('cum:', cumulative_probs)
    # Generate a random number between 0 and 1
    r = random.random()
    
    for i in range(len(fitnesses)):
        if r < cumulative_probs[i]:
            # print('+++++++++++++')
            # print(fitnesses[i])
            return i
    # Find which bin the random number falls into
    # for i, cumulative_prob in enumerate(cumulative_probs):
    #     if r <= cumulative_prob:
    #         print('**********')
    #         print(fitnesses[i])
    #         return i

def mutation1(population):
    print(population[2])
    for individual in population:
        for i in range(len(individual)):
            if random.random() < pm:
                individual[i] = random.choice([1, 2, 3, 4])
    print('&&&&&')
    print(population[2])


def mutation2(population):
    for individual in population:
        value = int(np.random.normal(mean, standard_deviation))
        if value > 0:
            for i in range(value):
                individual = np.append(individual, random.choice([1, 2, 3, 4]))
    


def find_crossover_points(location_individual1, location_individual2):
    crossover_points = []
    for i in range(len(location_individual1)):
        crossover_points.append([])
        for j in range(len(location_individual2)):
            if location_individual1[i] == location_individual2[j]:
                crossover_points[-1].append(j)
    return crossover_points

def choose_crossover_points(crossover_points):

    # Enumerate through the list to get both index and value
    non_empty_lists = [(index, non_empty) for index, non_empty in enumerate(crossover_points) if non_empty]

    if non_empty_lists:
        # Randomly choose from the filled lists
        index_parent1, choosed_non_empty_list = random.choice(non_empty_lists)
        index_parent2 = random.choice(choosed_non_empty_list)
    else:
        return None, None
    # print("Index1", index_parent1)
    # print("Index2", index_parent2)
    # print("Randomly chosen filled list:", choosed_non_empty_list)
    return index_parent1, index_parent2



def crossover(population, fitnesses, location_of_individuals_at_each_step):
    # for i in range(len(population)):
    #     print(population[i], (fitnesses[i] + abs(min(fitnesses))) * (fitnesses[i] + abs(min(fitnesses))) + 1)
        # print(population[i], fitnesses[i])
    selected_parent1 = roulette_wheel_selection(fitnesses)
    # print('test')
    selected_parent2 = roulette_wheel_selection(fitnesses)
    # print(selected_parent1)
    # print(selected_parent2)
    # print('---------')
    # test(population, fitnesses)
    # print("#####################")
    # print(population[2])
    # print(location_of_individuals_at_each_step[2])
    crossover_points = find_crossover_points(location_of_individuals_at_each_step[selected_parent1], location_of_individuals_at_each_step[selected_parent2])
    index_parent1, index_parent2 = choose_crossover_points(crossover_points)
    # print(crossover_points)
    if index_parent1 is None:
        return population[selected_parent1], population[selected_parent2]
    offspring1 = np.concatenate((population[selected_parent1][:index_parent1], population[selected_parent2][index_parent2:]))
    offspring2 = np.concatenate((population[selected_parent2][:index_parent2], population[selected_parent1][index_parent1:]))
    
    return offspring1, offspring2
    



if __name__ == '__main__':
    population = initialization(population_size)
    for i in range(number_of_iterations):
        fitnesses, location_of_individuals_at_each_step = fitness(population)
        new_population = []
        print('average of fitnesses in step ', i)
        print(sum(fitnesses)/ len(fitnesses))
        for i in range(population_size // 2):
            offspring1, offspring2 = crossover(population, fitnesses, location_of_individuals_at_each_step)
            new_population.append(offspring1)
            new_population.append(offspring2)
        mutation1(new_population)
        mutation2(new_population)
        population = copy.deepcopy(new_population)
    fitnesses, location_of_individuals_at_each_step = fitness(population)
    test(population, fitnesses)