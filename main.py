import math
import random
from timeit import default_timer as timer
from pyspark import SparkContext, SparkConf
import sys
import os


def MRFFT(inputPoints, K, sc):
    def local_fft(partition):
        points = list(partition)
        if points:

            return SequentialFFT(points, min(len(points), 50))
        else:
            return []

    local_centers_rdd = inputPoints.mapPartitions(local_fft)

    reduced_centers = local_centers_rdd.map(lambda center: (1, [center])).reduceByKey(lambda a, b: a + b)
    reduced_centers = reduced_centers.flatMap(lambda x: x[1]).collect()

    # Round 2: Compute final centers from the reduced set of local centers
    start2 = timer()
    centers = SequentialFFT(reduced_centers, K)
    end2 = timer()
    centers_broadcast = sc.broadcast(centers)
    print("Time for Round 2 (compute centers):", math.floor((end2 - start2) * 1000), "ms")

    # Round 3: Compute the radius of the clustering
    start3 = timer()

    def min_distance(point):
        return min(math.dist(point, center) for center in centers_broadcast.value)

    radius = inputPoints.map(min_distance).reduce(max)
    end3 = timer()
    print("Radius of clustering:", radius)
    print("Time for Round 3 (compute radius):", math.floor((end3 - start3) * 1000), "ms")
    return radius

def SequentialFFT(P,K):
    C = []
    C.append(random.choice(P))
    for i in range(0,K-1):
        max_dist = 0
        max_elem = 0
        for j in range(len(P)):
            if P[j] in C:
                continue
            distance = math.dist(C[i], P[j])
            for m in range(0, len(C)-1):
                new_distance = math.dist(C[m], P[j])
                if new_distance < distance:
                    distance = new_distance
            if distance > max_dist:
                max_dist = distance
                max_elem = P[j]
        C.append(max_elem)
    return C

def get_cell_id(x, y, Lambda):
    i = math.floor(x / Lambda)
    j = math.floor(y / Lambda)
    return (i, j)


# STEP A
def count_points_within_partition(iterator, Lambda):
    # empty dictionary for counting
    cell_counts = {}
    # iterate for each point of the partition
    for point in iterator:
        # calculates the cell ID for the current point based on its coordinates and lambda
        cell_id = get_cell_id(point[0], point[1], Lambda)
        # add cell id
        if cell_id not in cell_counts:
            cell_counts[cell_id] = 1
        else:
            cell_counts[cell_id] += 1
    # cell id and count as couple
    for cell_id, count in cell_counts.items():
        yield (cell_id, count)


# count the total points in each cell in all L partitions
def count_points_in_cells(points, Lambda):
    # uses RDD's mapPartitions method to apply count_points_within_partition to each partition
    local_cell_counts = points.mapPartitions(lambda it: count_points_within_partition(it, Lambda))
    # Reduce by key (cell_id) to do local counts
    # sums the counts for each cell ID across the entire dataset
    total_cell_counts = local_cell_counts.reduceByKey(lambda a, b: a + b)
    # returns the aggregate counts. Contains pairs of cell IDs and their total point count.
    return total_cell_counts


# STEP B
def extend_with_neighbour_counts(cell_counts, M):
    global cell, cell
    cells_dict = cell_counts.collectAsMap()
    extended_cells_info = []
    for cell_id, count in cells_dict.items():
        N3 = sum(cells_dict.get((cell_id[0] + dx, cell_id[1] + dy), 0) for dx in range(-1, 2) for dy in range(-1, 2))
        N7 = sum(cells_dict.get((cell_id[0] + dx, cell_id[1] + dy), 0) for dx in range(-3, 4) for dy in range(-3, 4))
        extended_cells_info.append((cell_id, count, N3, N7))
    sure_outliers = [cell for cell in extended_cells_info if cell[3] <= M]
    uncertain_points = [cell for cell in extended_cells_info if cell[2] <= M and cell[3] > M]
    out = 0
    for i in range(len(sure_outliers)):
        out += int(sure_outliers[i][1])
    print(f"Number of sure outliers  = ", out)
    unsure = 0
    for i in range(len(uncertain_points)):
        unsure += int(uncertain_points[i][1])
    print(f"Number of uncertain points = ", unsure)


'''MR ApproxOutliers manages to count points in L partitions and add N3 and N7, doing so in two steps.'''


def MRApproxOutliers(points_rdd, D, M):
    Lambda = D / (2 * math.sqrt(2))
    cell_counts = count_points_in_cells(points_rdd, Lambda)
    extend_with_neighbour_counts(cell_counts, M)

def main():
    assert len(sys.argv) == 5, "Incorrect number of parameters"
    print(sys.argv[1], "M=" + str(sys.argv[2]), "K=" + str(sys.argv[3]),"L=" + str(sys.argv[4]))
    data_path = sys.argv[1]
    assert os.path.isfile(data_path), "Incorrect data path"
    try:
        M = int(sys.argv[2])
        K = int(sys.argv[3])
        L = int(sys.argv[4])
    except:
        print("Incorrect type of parameters")
        exit(1)
    conf = SparkConf().setAppName('G020HW1')
    sc = SparkContext(conf=conf)
    rawData = sc.textFile(data_path).repartition(L).cache()
    inputPoints = rawData.map(lambda x: [float(i) for i in x.split(",")])
    print("Number of points =", inputPoints.count())
    D = MRFFT(inputPoints, K, sc)
    print("Computed radius (D) for MRApproxOutliers:", D)
    start1 = timer()
    MRApproxOutliers(inputPoints, D, M)
    end1 = timer()
    listofPoints = inputPoints.collect()
    print("Running time of MRApproxOutliers =", math.floor((end1 - start1) * 1000), "ms")


if __name__ == "__main__":
    main()
