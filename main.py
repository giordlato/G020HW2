import math
from timeit import default_timer as timer
from pyspark import SparkContext, SparkConf
import sys
import os

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


def MRFFT(inputPoints, K):
    return 0.02


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
    D = MRFFT(inputPoints,K)
    start1 = timer()
    MRApproxOutliers(inputPoints, D, M)
    end1 = timer()
    print("Running time of MRApproxOutliers =", math.floor((end1 - start1) * 1000), "ms")


if __name__ == "__main__":
    main()
