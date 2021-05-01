import numpy as np
import random

#params
FIELD_SIZE = 9
POPULATION_SIZE = 5000
SUCCESS_GENS = int(POPULATION_SIZE - POPULATION_SIZE / 5)
INITIAL_START = (5, 5)
MUTATE_PROB = 0.01
CROSS_OVER_POINT = 2
VERBOSE = 10
GENERATION = 200
DRONE_COUNT = 2
STEP_COUNT = int(FIELD_SIZE**2 / DRONE_COUNT)

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

def calculate_scanned_area_of_all_drones(drones, initial_start):
    mask = np.zeros(shape=(FIELD_SIZE, FIELD_SIZE), dtype=np.int)
    drone_paths = np.zeros(shape=(drones.shape[0], FIELD_SIZE, FIELD_SIZE), dtype=np.int)

    for index, drone in enumerate(drones):
        (x, y) = initial_start
        i = 0
        while i < drone.shape[0] and drone[i] != 0:
            (_x, _y) = directions[drone[i]]
            if 0 <= x + _x <= 8 and 0 <= y + _y <= 8:
                x += _x
                y += _y
                drone_paths[index][x][y] = 1
            i += 1

    for index in range(len(drones)):
        mask = np.logical_or(mask, drone_paths[index])

    print(f'scanned = {mask.sum()}')


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
def calculate_scanned_area(individual, initial_start, drones, drone_count):
    #best dronların pathini oluştur
    drone_paths = np.zeros(shape=(drone_count, FIELD_SIZE, FIELD_SIZE), dtype=np.int)
    (x, y) = initial_start
    for index, drone in enumerate(drones):
        i = 0
        while i < drone.shape[0] and drone[i] != 0 and index < drone_count:
            (_x, _y) = directions[drone[i]]
            if 0 <= x + _x <= 8 and 0 <= y + _y <= 8:
                x += _x
                y += _y
                drone_paths[index][x][y] = 1
            i += 1

    i = 0
    area = np.zeros(shape=(FIELD_SIZE, FIELD_SIZE), dtype=np.int32)
    (x, y) = initial_start
    area[x][y] = 1

    while i < individual.shape[0] and individual[i] != 0:
        (_x, _y) = directions[individual[i]]
        if 0 <= x + _x <= 8 and 0 <= y + _y <= 8:
            x += _x
            y += _y
            area[x][y] = 1
        i+=1

    (_x, _y) = initial_start
    start_and_stop_point_diff = np.sqrt(np.sum((x - _x)**2  + (y - _y)**2))

    common_path = np.zeros(shape=(FIELD_SIZE, FIELD_SIZE), dtype=np.int)
    for index in range(drone_count):
        common_path += drone_paths[index] * area

    return area.sum(), start_and_stop_point_diff, common_path.sum()


def fitness_function(population, initial_start, iteration, drones, drone_index):
    angles = np.zeros(shape=(population.shape[0], 1))
    areas  = np.zeros(shape=(population.shape[0], 1))
    correct_finish  = np.zeros(shape=(population.shape[0], 1))
    path_difference  = np.zeros(shape=(population.shape[0], 1))

    for index, individual in enumerate(population):
        areas[index], correct_finish[index], path_difference[index] = \
                                            calculate_scanned_area(individual, initial_start, drones, drone_index)
        angles[index] = calculate_sum_angle(individual)

    areas = STEP_COUNT - areas
    areas /= STEP_COUNT #azalacağız
    angles /= (4 * STEP_COUNT)  #azaltacağız
    correct_finish /= FIELD_SIZE * np.sqrt(2)   #azaltscağız

    path_difference = (STEP_COUNT - path_difference) / STEP_COUNT

    probabilities = areas + angles + correct_finish + path_difference
    probabilities /= probabilities.sum()

    probabilities = 1 - probabilities
    probabilities /= probabilities.sum()

    if iteration % VERBOSE == 0:
        print(f'Best gene\nunscanned area :\n {areas[np.argmax(probabilities)].T * STEP_COUNT}\n'
              f'angles : \n{angles[np.argmax(probabilities)].T * (4 * STEP_COUNT)}\n')
        print_path(population[np.argmax(probabilities)])

    return probabilities * 100


def cross_over(X, Y):
    if CROSS_OVER_POINT == 1:
        index = np.random.randint(0, X.shape[1])
        if np.random.rand() > 0.5:
            return np.append(X[:, 0 : index], Y[:, index:])
        else:
            return np.append(Y[:, 0: index], X[:, index:])
    else:
        index   = np.random.randint(0, X.shape[1] / 2)
        index_2 = np.random.randint(X.shape[1] / 2, X.shape[1])
        if np.random.rand() > 0.5:
            return np.append(np.append(X[:, 0 : index], Y[:, index: index_2]), X[:, index_2 :])
        else:
            return np.append(np.append(Y[:, 0: index], X[:, index:index_2]), Y[:, index_2 :])


def mutate(child):
    child[random.randint(0, child.shape[0] - 1)] = random.randint(1, 8)
    return child


def selection(population, population_probabilities):
    index = random.choices(np.arange(0, population.shape[0]), weights=population_probabilities)
    return population[index]


def start_field_scanning(drone_count, population_size, initial_start):
    drones = np.zeros(shape=(drone_count, STEP_COUNT), dtype=np.int)

    for drone_index in range(drone_count):
        # init population
        population = np.random.randint(low=1, high=9, size=(population_size, STEP_COUNT))
        new_population = np.zeros(shape=(population_size, STEP_COUNT), dtype=np.int32)
        best_individuals = np.ones(shape=(GENERATION, STEP_COUNT), dtype=np.int32)
        best_individual_probability = -1
        generation = 0

        while generation < GENERATION:
            generation += 1
            if generation % 10 == 0:
                print(f'iter : {generation}')

            population_probabilities = fitness_function(population, initial_start, generation, drones, drone_index)

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

            best_individuals[generation - 1] = success_population[-1]

            new_population[SUCCESS_GENS:, :] = success_population[SUCCESS_GENS:, :]
            population = new_population

        #we got all successful individuals in every generation
        success_population_prob = fitness_function(best_individuals, initial_start, generation, drones, drone_index)
        success = np.argsort(success_population_prob, axis=0)
        success_population = np.take_along_axis(best_individuals, success, 0)

        drones[drone_index] = success_population[-1]

    print('Done!')
    for index in range(len(drones)):
        print(f'Birey : {index}\n')
        print_path(drones[index])

    calculate_scanned_area_of_all_drones(drones, initial_start)


if __name__ == '__main__':
    start_field_scanning(DRONE_COUNT, POPULATION_SIZE, INITIAL_START)