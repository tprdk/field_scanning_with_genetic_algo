import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image



#params
FIELD_SIZE = 9# taranacak alanın satır ve sütun sayısı
POPULATION_SIZE = 1000# popülasyon büyüklüğü
SUCCESS_GENS = int(POPULATION_SIZE - POPULATION_SIZE / 5)   # bir sonraki jenerasyona aktarılacak birey sayısı
INITIAL_START = (8, 4)  # başlangıç noktası
MUTATE_PROB = 0.01  # mutasyon oranı
CROSS_OVER_POINT = 2    # crossover yapılacak nokta sayısı
VERBOSE = 5000
GENERATION = 200    # jenerasyon sayısı
DRONE_COUNT = 2# drone sayısı
STEP_COUNT = int((FIELD_SIZE**2 - 1) / DRONE_COUNT) # her bir bireyin adım sayısı

colors = ['blue', 'green', 'lime', 'cyan']  #plot renkleri
plt.rcParams.update({'font.size': 18})
np.warnings.filterwarnings('ignore')

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

# gidilecek yönlerin matris karşılığı
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


def print_final_path():
    image = Image.open(f'plot/p{POPULATION_SIZE}_g{GENERATION}_m{MUTATE_PROB}_d{DRONE_COUNT}_path.png')
    image.show()


def print_all_statistics(best_individuals, initial_start, drones, drone_index):
    '''
    tüm jenerasyonların en başarılı bireyleri için taranan alan, yapılan açı,
    başlangıç-bitiş mesafesi, önceki dronelar ile taranan ortak alan büyüklüğü ve
    iyilik değerlerinin değişimini gösteren grafikler bastırılır.
    :param best_individuals: tüm jenerasyonların en başarılı bireyleri
    :param initial_start:
    :param drones:
    :param drone_index:
    :return:
    '''
    areas = np.zeros(shape=(best_individuals.shape[0], 1))
    correct_finish = np.zeros(shape=(best_individuals.shape[0], 1))
    path_difference = np.zeros(shape=(best_individuals.shape[0], 1))
    angles = np.zeros(shape=(best_individuals.shape[0], 1))

    for index, individual in enumerate(best_individuals):
        areas[index], correct_finish[index], path_difference[index] = \
            calculate_scanned_area(individual, initial_start, drones, drone_index)
        angles[index] = calculate_sum_angle(individual)



    fig = plt.figure()
    fig.suptitle(f'Populasyon : {POPULATION_SIZE} Jenerasyon : {GENERATION} '
                 f'Mutasyon Oranı : {MUTATE_PROB} Crossover Noktası : {CROSS_OVER_POINT} '
                 f'\nMax Drone Hareketi : {STEP_COUNT} Başlangıç Noktası : {INITIAL_START}'
                 f' Toplam Drone Sayısı : {DRONE_COUNT} Drone index : {drone_index + 1} ')
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    #ax5 = fig.add_subplot(325)

    ax1.title.set_text('Taranan Alan')
    ax2.title.set_text('Yapılan Açı')
    ax3.title.set_text('Başlangıca Olan Uzaklık')
    ax4.title.set_text('Diğer Dronelar ile ortak yol')
    #ax4.title.set_text('Prob')

    ax1.plot(areas)
    ax2.plot(angles * 45)
    ax3.plot(correct_finish)
    ax4.plot(path_difference)

    areas = STEP_COUNT - areas
    # gezilmemiş alan
    areas /= STEP_COUNT  # azaltacağız
    # gezilen açı
    angles /= (4 * STEP_COUNT)  # azaltacağız
    # uzaklık
    correct_finish /= FIELD_SIZE * np.sqrt(2)  # azaltscağız
    # ortak yol
    path_difference /= STEP_COUNT

    probabilities = areas + angles + correct_finish + path_difference
    probabilities /= max(probabilities)

    probabilities = 1 - probabilities
    probabilities /= max(probabilities)
    probabilities *= 100

    #ax4.plot(probabilities)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    #plt.show()
    plt.savefig(f'plot/p{POPULATION_SIZE}_g{GENERATION}_m{MUTATE_PROB}_d{DRONE_COUNT}_{drone_index + 1}_fitness.png')
    plt.close(fig)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(probabilities)
    ax1.title.set_text('Jenerasyonların İyilikleri')
    plt.savefig(f'plot/p{POPULATION_SIZE}_g{GENERATION}_m{MUTATE_PROB}_d{DRONE_COUNT}_{drone_index + 1}_prob.png')
    plt.close(fig)
    print('saved')


def calculate_scanned_area_of_all_drones(drones, initial_start):
    '''
    bütün jenerasyonların tamamlanmasının ardından toplam taranan yol hesaplanır ve png formatında kaydedilir.
    :param drones:
    :param initial_start:
    :return:
    '''
    mask = np.zeros(shape=(FIELD_SIZE, FIELD_SIZE), dtype=np.int32)
    drone_paths = np.zeros(shape=(drones.shape[0], FIELD_SIZE, FIELD_SIZE), dtype=np.int32)
    drone_paths_plot = np.zeros(shape=(drones.shape[0], drones.shape[1] + 1, 2), dtype=np.int32)

    coordinate_indexes = np.zeros(shape=(drones.shape[0]), dtype='int')
    for index, drone in enumerate(drones):
        (x, y) = initial_start
        i = 0
        drone_paths_plot[index][coordinate_indexes[index]] = (x, y)
        while i < drone.shape[0]:
            (_x, _y) = directions[drone[i]]
            if 0 <= x + _x <= 8 and 0 <= y + _y <= 8:
                x += _x
                y += _y
                drone_paths[index][x][y] = 1
                drone_paths_plot[index][coordinate_indexes[index] + 1] = (x, y)
                coordinate_indexes[index] += 1
            i += 1

    for index in range(len(drones)):
        mask = np.logical_or(mask, drone_paths[index])

    print(f'scanned = {mask.sum()}')

    plt.xticks(np.arange(-1, 9, 1))
    plt.yticks(np.arange(-1, 9, 1))
    plt.xlim(-1, 9)
    plt.ylim(-1, 9)

    for i in range(drone_paths_plot.shape[0]):
        x = drone_paths_plot[i][0:coordinate_indexes[i] + 1, 0]
        y = drone_paths_plot[i][0:coordinate_indexes[i] + 1, 1]
        plt.plot(y, x, '-', color=colors[i])
        plt.plot(y[-1], x[-1], 'ro', color=colors[i])

    x, y = initial_start
    plt.plot(y, x, 'ro', color='black')
    plt.title(f'Taranan Toplam Alan : {mask.sum()}')
    plt.gca().invert_yaxis()
    plt.savefig(f'plot/p{POPULATION_SIZE}_g{GENERATION}_m{MUTATE_PROB}_d{DRONE_COUNT}_path.png')
    plt.close()


def print_path(path):
    '''
    fonksiyona gönderilen bireyin yolunun bastırılması
    :param path: birey
    '''
    i = 0
    (x, y) = INITIAL_START
    area = np.zeros(shape=(FIELD_SIZE, FIELD_SIZE), dtype=np.int8)
    area[x][y] = 1
    while i < path.shape[0]:
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
    '''
    Bireyin yaptığı açı bulunur.
    :param individual: birey
    :return: angle: yapılan toplam açı
    '''

    i = 0
    angle = 0
    while i + 1 < individual.shape[0]:
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


def calculate_scanned_area(individual, initial_start, drones, drone_index):
    '''
    :param individual: birey
    :param initial_start: başlangıç noktası
    :param drones: drone listesi
    :param drone_index: şu an yolu bulunacak olan drone indexi

    :return: area.sum(): bireyin taradığı toplam alan
    :return: start_and_stop_point_diff: bireyin başlangıç ve bitiş noktası arası uzaklığı
    :return: common_path.sum(): birey ile listedeki drone'ların taradığı ortak alan
    '''

    #listedeki dronların yolu bulunur
    drone_paths = np.zeros(shape=(drone_index, FIELD_SIZE, FIELD_SIZE), dtype=np.int32)
    (x, y) = initial_start
    for index, drone in enumerate(drones):
        i = 0
        while i < drone.shape[0] and index < drone_index:
            (_x, _y) = directions[drone[i]]
            if 0 <= x + _x <= 8 and 0 <= y + _y <= 8:
                x += _x
                y += _y
                drone_paths[index][x][y] = 1
            i += 1

    #bireyin yolu bulunur
    i = 0
    area = np.zeros(shape=(FIELD_SIZE, FIELD_SIZE), dtype=np.int32)
    (x, y) = initial_start
    area[x][y] = 1

    while i < individual.shape[0]:
        (_x, _y) = directions[individual[i]]
        if 0 <= x + _x <= 8 and 0 <= y + _y <= 8:
            x += _x
            y += _y
            area[x][y] = 1
        i+=1

    #son konum ile ilk konum arasındaki uzaklık bulunur
    (_x, _y) = initial_start
    start_and_stop_point_diff = np.sqrt(np.sum((x - _x)**2  + (y - _y)**2))

    #bireyin yolu ile listedeki dronların ortak olarak taradıkları alan bulunur
    common_path = np.zeros(shape=(FIELD_SIZE, FIELD_SIZE), dtype=np.int32)
    for index in range(drone_index):
        common_path += drone_paths[index] * area

    return area.sum(), start_and_stop_point_diff, common_path.sum()


def fitness_function(population, initial_start, iteration, drones, drone_index):
    '''
    :param population: popülasyon
    :param initial_start: başlangıç noktası
    :param iteration: popülasyon indexi
    :param drones:  drone listesi
    :param drone_index: şu an yolu bulunacak olan drone indexi

    :return: probabilities: bireylerin iyilikleri
    '''

    angles = np.zeros(shape=(population.shape[0], 1))
    areas  = np.zeros(shape=(population.shape[0], 1))
    correct_finish  = np.zeros(shape=(population.shape[0], 1))
    path_difference  = np.zeros(shape=(population.shape[0], 1))

    #popülasyon içerisindeki her birey için :
    #taranan alan, başlangıç-bitiş mesafesi, önceki dronelar ile gidilen ortak yol ve toplam yapılan açı bulunur.
    for index, individual in enumerate(population):
        areas[index], correct_finish[index], path_difference[index] = \
                                            calculate_scanned_area(individual, initial_start, drones, drone_index)
        angles[index] = calculate_sum_angle(individual)

    #bulunan değerler min-max normalizasyonu ile normalize edilir.
    #areas dizisi taranan alanları tutmaktadır. toplam adım sayısından taranan alan çıkartılarak
    #taranmamış alan bulunur.
    areas = STEP_COUNT - areas
    areas /= STEP_COUNT

    #angles dizisi içerisinde yapılan açılar tutulur
    angles /= (4 * STEP_COUNT)

    #correct_finish dizisinde başlangıç-bitiş mesafesi tutulur
    correct_finish /= FIELD_SIZE * np.sqrt(2)

    #path_difference dizisinde önceki drone'lar ile gidilen ortak yol tutulur.
    path_difference /= STEP_COUNT

    #iyilik fonksiyonunu hesaplamak için normalize edilen 4 dizi toplanır ve normalize edilir.
    probabilities = areas + angles + correct_finish + path_difference
    probabilities /= max(probabilities)

    #4 dizinin değerlerini azaltmak istediğimiz için tümleyeni alınır ve normalize edilir
    #en başarılı birey en yüksek iyiliğe sahip olacak hale getirilir
    probabilities = 1 - probabilities
    if max(probabilities) != 0:
        probabilities /= max(probabilities)
    else:
        probabilities += 1

    if iteration % VERBOSE == 0:
        print(f'Best gene\nunscanned area :\n {areas[np.argmax(probabilities)].T * STEP_COUNT}\n'
              f'angles : \n{angles[np.argmax(probabilities)].T * (4 * STEP_COUNT)}\n')
        print_path(population[np.argmax(probabilities)])
        print(np.round(time.time() - start_time, 4))

    return probabilities * 100


def cross_over(X, Y):
    '''
    fonksiyon gelen 2 bireyin crossover noktası kadar noktadan parça değiştirmesini sağlar
    crossover noktası kadar random indis üretilerek o noktalardan yeni dizi oluşturulur.
    :param X: birey 1
    :param Y: birey 2
    :return: yeni birey
    '''
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
    '''
    gelen bireyin random bir indisi random bir hareket ile değiştirilir
    :param child: birey
    :return: child: mutasyona uğramış birey
    '''
    child[random.randint(0, child.shape[0] - 1)] = random.randint(1, 8)
    return child


def selection(population, population_probabilities):
    '''
    Rulet tekeri uygulanarak bir bireyin seçilmesi sağlanır.
    popülasyonun iyilik değerleri üzerinden bir ağırlıklı seçim yapılır.
    :param population: popülasyon
    :param population_probabilities: popülasyonun iyilik durumları
    :return: population[index] : seçilen birey
    '''
    index = random.choices(np.arange(0, population.shape[0]), weights=population_probabilities)
    return population[index]


def start_field_scanning(drone_count, population_size, initial_start):
    '''
    tüm dronelar için genetik algoritma çalıştırılır
    :param drone_count: alanı tarayacak drone sayısı
    :param population_size: popülasyon büyüklüğü
    :param initial_start: başlangıç noktası
    '''
    drones = np.zeros(shape=(drone_count, STEP_COUNT), dtype=np.int32)
    worst_drones = np.zeros(shape=(drone_count, STEP_COUNT), dtype=np.int32)

    for drone_index in range(drone_count):

        # ilk popülasyon random sayılar ile oluşturulur
        population = np.random.randint(low=1, high=9, size=(population_size, STEP_COUNT))
        new_population = np.zeros(shape=(population_size, STEP_COUNT), dtype=np.int32)
        best_individuals = np.ones(shape=(GENERATION, STEP_COUNT), dtype=np.int32)
        worst_individuals = np.ones(shape=(GENERATION, STEP_COUNT), dtype=np.int32)

        generation = 0
        while generation < GENERATION:
            generation += 1
            if generation % 10 == 0:
                print(f'drone : {drone_index + 1} - generation : {generation}')

            # popülasyondaki her birey için iyilik değeri hesaplanır
            population_probabilities = fitness_function(population, initial_start, generation, drones, drone_index)

            for i in range(population.shape[0]):
                # iyilik değerleriyle ağırlıklandırılmış olasılıklarla 2 birey seçilir
                X = selection(population, population_probabilities)
                Y = selection(population, population_probabilities)
                # belirlenen nokta sayısı ile 2 birey çaprazlanır
                child = cross_over(X, Y)
                # belirlenen mutasyon oranı ile birey mutasyona uğratılır
                if MUTATE_PROB > np.random.rand():
                    child = mutate(child)

                # oluşturulan birey yeni popülasyona eklenir
                new_population[i] = child

            # popülasyonun iyilik değerlerine göre indisler küçükten büyüğe sıralanır
            success = np.argsort(population_probabilities, axis=0)
            success_population = np.take_along_axis(population, success, 0)
            # popülasyonun en başarılı bireyi, her jenerasyonun en iyi bireyini tutan diziye yazılır
            best_individuals[generation - 1] = success_population[-1]
            worst_individuals[generation - 1] = success_population[0]

            # belirlenen sayıda en başarılı birey, yeni popülasyona kopyalanır
            new_population[SUCCESS_GENS:, :] = success_population[SUCCESS_GENS:, :]
            population = new_population

        #tüm jenerasyonlardaki en başarılı bireylerin grafikleri bastırılır
        print_all_statistics(best_individuals, initial_start, drones, drone_index)

        #tüm jenerasyonlardaki başarılı bireyler arasından en başarılı birey seçilerek drone listesine eklenir
        success_population_prob = fitness_function(best_individuals, initial_start, generation, drones, drone_index)
        success = np.argsort(success_population_prob, axis=0)
        success_population = np.take_along_axis(best_individuals, success, 0)

        drones[drone_index] = success_population[-1]
        worst_drones[drone_index] = success_population[0]

    calculate_scanned_area_of_all_drones(drones, initial_start)
    print_final_path()

if __name__ == '__main__':
    start_time = time.time()
    start_field_scanning(DRONE_COUNT, POPULATION_SIZE, INITIAL_START)
