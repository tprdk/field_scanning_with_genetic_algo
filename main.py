import numpy as np
import random

#params
FIELD_SIZE = 9
POPULATION_SIZE = 500
SUCCESS_GENS = int(POPULATION_SIZE - POPULATION_SIZE / 5)
INITIAL_START = (8, 0)
MUTATE_PROB = 0.1
VERBOSE = 20
EPOCHS = 200
STEP_COUNT = 40


# 2 1 8
# 3 0 7
# 4 5 6

directions = {
    1 : (-1, 0),
    2 : (-1, -1),
    3 : (0, -1),
    4 : (1, -1),
    5 : (1, 0),
    6 : (1, 1),
    7 : (0, 1),
    8 : (-1, 1)
}

def print_path(path):
    i = 0
    (x, y) = INITIAL_START
    area = np.zeros(shape=(FIELD_SIZE, FIELD_SIZE), dtype=np.int8)
    area[x][y] = 1
    while i + 1 < path.shape[0] and path[i + 1] != 0:
        (_x, _y) = directions[path[i]]
        if 0 <= x + _x <= 8 and 0 <= y + _y <= 8:
            x += _x
            y += _y
            area[x][y] = i + 2
        i += 1

    for i in range(FIELD_SIZE):
        for j in range(FIELD_SIZE):
            print(f'{area[i][j]:>3}', end=' ')
        print()


def fitness_function(population, initial_start, iter):
    sum_angles = 0
    sum_unscanned_area = 0
    angles = np.zeros(shape=(population.shape[0], 1))
    areas  = np.zeros(shape=(population.shape[0], 1))
    probabilities = np.zeros(shape=(1, population.shape[0]))

    for index, individual in enumerate(population):
        areas[index] = calculate_scanned_area(individual, initial_start)
        angles[index] = calculate_sum_angle(individual)

    if iter % VERBOSE == 0:
        print(f'area :\n {areas.T} - angles : \n{angles.T}\n')

    areas /= 81
    angles /= (4 * 80)

    probabilities = areas + (1 /angles)
    probabilities /= probabilities.sum()
    return probabilities * 100

    # 2 1 8
    # 3 0 7
    # 4 5 6

    # 3 - 1  -> 2
    # 3 - 2  -> 1
    # 3 - 3  -> 0
    # 3 - 4  -> 1
    # 3 - 5  -> 2
    # 3 - 6  -> 3
    # 3 - 7  -> 4
    # 3 - 8  -> 3

    # 7 - 1  -> 2
    # 7 - 2  -> 3
    # 7 - 3  -> 4
    # 7 - 4  -> 3
    # 7 - 5  -> 2
    # 7 - 6  -> 1
    # 7 - 7  -> 0
    # 7 - 8  -> 1

    # 8 - 1  -> 1
    # 8 - 2  -> 2
    # 8 - 3  -> 3
    # 8 - 4  -> 4
    # 8 - 5  -> 3
    # 8 - 6  -> 2
    # 8 - 7  -> 1
    # 8 - 8  -> 0

def calculate_sum_angle(individual):
    i = 0
    angle = 0
    # 4 4 5 6 1 2 3 8 7 4 5 6
    while i + 1 < individual.shape[0] and individual[i + 1] != 0:
        current = individual[i]
        _next = individual[i + 1]
        _max = max(current, _next)
        _min = min(current, _next)
        if _max - _min <= 4:
            angle += _max - _min
        else:
            angle += (8 + _min - _max)
        i += 1
    return angle

    # 2 1 8
    # 3 0 7
    # 4 5 6
def calculate_scanned_area(individual, initial_start):
    i = 0
    area = np.zeros(shape=(FIELD_SIZE, FIELD_SIZE), dtype=np.int32)
    (x, y) = initial_start
    area[x][y] = 1
    #8 0
    # 4 4 5 6 1 2 3 8 7 4 5 6
    #aynı yere gidiyor
    while i < individual.shape[0] and individual[i] != 0:
        (_x, _y) = directions[individual[i]]
        if 0 <= x + _x <= 8 and 0 <= y + _y <= 8:
            x += _x
            y += _y
            area[x][y] = 1
        i+=1
    return area.sum()

def cross_over(X, Y):
    index = np.random.randint(0, X.shape[1])
    if np.random.rand() > 0.5:
        return np.append(X[:, 0 : index], Y[:, index:])
    else:
        return np.append(Y[:, 0: index], X[:, index:])


def mutate(child):
    child[random.randint(0, child.shape[0] - 1)] = random.randint(1, 8)
    return child


def selection(population, population_probabilities):
    index = random.choices(np.arange(0, population.shape[0]), weights=population_probabilities)
    return population[index]

def start_field_scanning(field, population_size, initial_start):
    #init population
    population = np.random.randint(low=1, high=9, size=(population_size, field.shape[0] * field.shape[1]))
    new_population = np.zeros(shape=(population_size, field.shape[0] * field.shape[1]), dtype=np.int32)

    j = 0
    while j < EPOCHS:
        j += 1
        if j % VERBOSE == 0:
            for i in range(population.shape[0]):
                print(f'Birey {i} :')
                print(population[i])
                print_path(population[i])
        if j % 10 == 0:
            print(f'iter : {j}')

        population_probabilities = fitness_function(population, initial_start, j)

        #loop through population
        for i in range(population.shape[0]):
            X = selection(population, population_probabilities)
            Y = selection(population, population_probabilities)
            child = cross_over(X, Y)

            if MUTATE_PROB > np.random.rand():
                child = mutate(child)
            new_population[i] = child

        #get most successful indexes
        success = np.argsort(population_probabilities, axis=0)
        success_population = np.take_along_axis(population, success, 0)

        new_population[SUCCESS_GENS:, :] = success_population[SUCCESS_GENS:, :]
        population = new_population


if __name__ == '__main__':
    start_field_scanning(np.zeros(shape=(FIELD_SIZE, FIELD_SIZE), dtype=np.int32), POPULATION_SIZE,
                         INITIAL_START)