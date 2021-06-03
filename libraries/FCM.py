import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

class FCM():
    N_CLUSTER = 5
    MIN_ERROR = 0.01
    MAX_ITERATION = 100
    PREV_TOTAL = 0

    data = None
    center = None
    label = ""
    y = None
    c_cluster = []

    def __init__(self, n_cluster, min_error, max_iteration):
        self.N_CLUSTER = n_cluster
        self.MIN_ERROR = min_error
        self.MAX_ITERATION = max_iteration

    def read_csv(self, path: str, label: str, cols: list):
        df = pd.read_csv(path)
        self.label = label
        self.data = df[cols].values
        self.y = df[label].values

        del df

        # print(self.data)
    
    def generate_random(self):
        np.random.seed(1)
        self.center = np.random.dirichlet(np.ones(self.N_CLUSTER),size=len(self.data))
        # print(self.c)

    def show_result(self, show_all=False, group_by=False):

        # cluster = ["C1", "C2", "C3", "C4", "C5"]
        nc = {}
        nc["label"] = self.y

        arr = []
        
        for c in self.center:
            arr.append(np.where(c == np.amax(c))[0][0] + 1)
            
        arr = np.array(arr)

        nc["cluster"] = arr

        self.result = pd.DataFrame(nc)

        pd.set_option('max_rows', None if show_all else 5)
        print(self.result.head(187))

        if group_by:
            print(self.result.groupby(["cluster"]).agg({"cluster": ["count"]}))



    def step_1(self):
        c_2 = np.power(self.center, 2)

        cluster = ["C1", "C2", "C3", "C4", "C5"]

        obj = {}

        sum = []

        sum_c2 = []

        for j, c in enumerate(cluster):
            arr = []
            for i in range(len(self.data)):
                arr.append(c_2[i, j] * self.data[i, :])
            obj[c] = np.array(arr)
            sum_c2.append(np.sum(c_2[:, j]))
            sum.append([])

            for i in range(len(self.data[0])):
                
                sum[j].append(np.sum(obj[c][:, i]))

        sum_c2 = np.array(sum_c2)
        sum = np.array(sum)

        self.c_cluster = []

        for i, sc2 in enumerate(sum_c2):
            self.c_cluster.append([])
            for j in range(len(sum[0])):
                self.c_cluster[i].append(sum[i, j] / sc2)

        self.c_cluster = np.array(self.c_cluster)
                
        # print(self.c_cluster)

    def step_2(self):

        cluster = []

        c_2 = np.power(self.center, 2)

        L = []

        for i, c in enumerate(self.c_cluster):
            cluster.append(np.power(self.data - c, 2))

        cluster = np.array(cluster)

        total_l = []

        matrix_u = []

        total_matrix_u = []

        # obj function and matrix partition u

        for i, row in enumerate(c_2):
            L.append([])
            matrix_u.append([])
            for j, col in enumerate(row):
                L[i].append(np.sum(cluster[j][i, :]) * c_2[i, j])
                matrix_u[i].append(np.power(np.sum(cluster[j][i, :]), -1))
            total_l.append(np.sum(L[i]))
            total_matrix_u.append(np.sum(matrix_u[i]))

        matrix_u = np.array(matrix_u)

        total_matrix_u = np.array(total_matrix_u)
            
        total_l = np.array(total_l)
        L = np.array(L)

        total = np.sum(total_l)

        error = total - self.PREV_TOTAL

        self.PREV_TOTAL = total

        # update center

        new_center = []

        for i, row in enumerate(matrix_u):
            new_center.append(row / total_matrix_u[i])

        self.center = np.array(new_center)

        # print(new_center)

        return np.abs(error), total

    def start(self, verbose=False):

        for i in range(self.MAX_ITERATION):
            self.step_1()
            error, total = self.step_2()

            if verbose:
                print("error====>", error)
                print("total====>", total)
                print("epoch====>", i+1)
                print()
            
            if error < self.MIN_ERROR:
                break

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = self.data[:, 0]
        y = self.data[:, 1]
        z = self.data[:, 2]

        ax.scatter(x,y,z, linewidths=0.5, alpha=.7, edgecolor='k', marker="s", c=self.result["cluster"], s=40)

        plt.show()
