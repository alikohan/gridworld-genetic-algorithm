# import numpy as np

# arr = np.array([4, 2, 7, 1, 9])

# # Get the indices that would sort the array
# indices = np.argsort(arr)

# print("Original Array:", arr)
# print("Indices that would sort the array:", indices)
# print((2 - indices) / (2 * 5))
import numpy as np
import random

# Define your list of fitnesses
result = []
fitnesses = [2, 4, 3, 7, -13, -2]
for i in range(10000):

    # Given list of fitnesses

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
    selected_index
    result.append(selected_index)
for i in range(len(fitnesses)):
    print(result.count(i))
