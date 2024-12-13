from itertools import permutations
import json

inner_layers = [1, 2, 4, 8, 16, 32]

# Generate all permutations of lengths 1 to len(nums)
all_permutations = [json.dumps(list(p)) for r in range(1, len(inner_layers) + 1) for p in permutations(inner_layers, r)]

for permutation in all_permutations:
    print(permutation)
print(f"Total permutations: {len(all_permutations)}")


fjdkfdj = 1
