# import random

# # Your list containing empty and filled lists
# your_list = [[], [1, 2, 3], [], [4, 5, 6], [], [7, 8, 9], [11, 12, 13], [7, 8, 9]]

# # Enumerate through the list to get both index and value
# filled_lists = [(index, lst) for index, lst in enumerate(your_list) if lst]
# print(type(filled_lists[0]))
# # Randomly choose from the filled lists
# index, random_filled_list = random.choice(filled_lists)

# print("Index of the chosen filled list:", index)
# print("Randomly chosen filled list:", random_filled_list)
import numpy as np
import matplotlib.pyplot as plt

mean, standard_deviation = 0, 1 # mean and standard deviation
numbers = []
for i in range(10000):
    numbers.append(int(np.random.normal(mean, standard_deviation)))
# print(s)
# Plot the histogram
plt.hist(numbers, bins=100, edgecolor='black', alpha=0.7)
print(numbers[0:100])

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of List of Numbers')

# Show the plot
plt.show()