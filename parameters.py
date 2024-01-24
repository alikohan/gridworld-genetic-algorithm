# 1 represents an obstacle, while 0 represents a free state
environment = [['A',0, 0, 1, 0], 
                [0, 0, 0, 0, 0], 
                [0, 0, 1, 1, 0],
                [0, 0, 1, 'T', 0], 
                [0, 0, 0, 0, 1]]

# environment = [['A',0, 0, 1], 
#                 [0, 0, 0, 0], 
#                 [0, 0, 1, 0],
#                 [0, 0, 1, 'T']]

agent_row = 0 # the position of agent is in which row
agent_column = 0 # the position of agent is in which column
target_row = 3
target_column = 3
population_size = 100
pm = 0.1
number_of_iterations = 100
length_of_individual_initialization = 8
mean, standard_deviation = 0, 1 # for mutation 2
importance_of_shortest_length = 0.2 # the priority of finding short answers (it impacts the convergence, change it carefully!)
