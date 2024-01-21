import random

def roulette_wheel_selection(fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f/total_fitness for f in fitnesses]
    
    # Create a cumulative sum of the selection probabilities
    cumulative_probs = [sum(selection_probs[:i+1]) for i in range(len(selection_probs))]
    
    # Generate a random number between 0 and 1
    r = random.random()
    
    # Find which bin the random number falls into
    for i, cumulative_prob in enumerate(cumulative_probs):
        if r <= cumulative_prob:
            return i

# Given list of fitnesses
fitnesses = [2, 4, -3, 7, -12, 3, 14, -126]
result = []
# Perform roulette wheel selection
for i in range(1000):
    result.append(roulette_wheel_selection(fitnesses))

for i in range(len(fitnesses)):
    print(result.count(i))
# Output the selected index
# print(selected_index)