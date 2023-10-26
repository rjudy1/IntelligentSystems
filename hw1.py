import math
import random


def change_letters(query, letters_to_change):
    letters_changed = 0
    query_array = list(query)
    while letters_changed < letters_to_change:
        index = random.randrange(0, len(query))
        replacement = chr(random.randrange(ord('a'), ord('z') + 1))
        if query_array[index] != ' ' and index != len(query) - 1\
                and query_array[index] == query[index] and query[index].lower() != replacement:
            query_array[index] = replacement
            letters_changed += 1
    return ''.join(query_array)


if __name__ == "__main__":
    initial_query = "What are the great pyramids?"
    print(f'query for epoch 0, trial 0: {initial_query}')

    for epoch in range(1, 7):
        for trial in range(7):
            new_query = change_letters(initial_query, math.ceil(epoch * .1 * (len(initial_query) - 5)))
            print(f'{epoch},{trial},{new_query},')
