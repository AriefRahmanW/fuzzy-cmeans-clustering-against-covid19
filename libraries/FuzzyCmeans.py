import pandas as pd
import numpy as np

class FuzzyCmeans():
    MAX_CLUSTER = 5
    MIN_ERROR = 0.01
    MAX_ITERATION = 100
    PREV_TOTAL = 0
    df = None
    c_df = None
    center_cluster = None
    all_cluster = {}
    cluster = {}
    obj_function = None
    matrix_partition = None

    def __init__(self, max_cluster, min_error, max_iteration):
        self.MAX_CLUSTER = max_cluster
        self.MIN_ERROR = min_error
        self.MAX_ITERATION = max_iteration

    def select_data(self, path: str, label: str, cols: list):
        self.df = pd.read_csv(path)
        self.label = label
        self.cols = self.df[cols]

        self.init_var()
        self.init_random()

    def show_data(self):
        print(self.cols.head())

    def generate_random(self):
        np.random.seed(0)
        return np.random.dirichlet(np.ones(self.MAX_CLUSTER),size=len(self.cols))

    def init_var(self):
        self.c_colname = ["C" + str(i + 1) for i in range(self.MAX_CLUSTER)]
        self.x_colname = ["X" + str(i + 1) for i in range(len(self.cols.columns))]
        self.v_colname = ["V" + str(i + 1) for i in range(self.MAX_CLUSTER)]
        self.l_colname = ["L" + str(i + 1) for i in range(self.MAX_CLUSTER)]
        self.cluster_name = ["CLUSTER_" + str(i + 1) for i in range(self.MAX_CLUSTER)]

        self.cols.columns = self.x_colname

    def init_random(self):
        self.c_df = pd.DataFrame(self.generate_random())
        self.c_df.columns = self.c_colname
    
    def show_random(self):
        print(self.c_df)

    def show_result(self, show_all=False):
        result = pd.DataFrame()
        result["label"] = self.df[self.label]
        for i, c in enumerate(self.c_colname):
            result[self.cluster_name[i]] = np.where(self.c_df[c] == self.c_df.max(axis=1), "1", "")

        pd.set_option('max_rows', None if show_all else 5)
        print(result.head(187))

        del result
    
    def start_step_1(self):
        self.all_cluster = {}

        div = []

        for i, c in enumerate(self.cluster_name):
            self.all_cluster[c] = {}
            self.all_cluster[c]["DATA"] = pd.DataFrame()
            
            self.all_cluster[c]["DATA"][self.c_colname[i] + "^2"] = np.power(self.c_df[self.c_colname[i]], 2)
            self.all_cluster[c]["SUM_" + self.c_colname[i] + "^2"] = np.sum(self.all_cluster[c]["DATA"][self.c_colname[i] + "^2"])
            div.append([])
            for j, x in enumerate(self.x_colname):
                self.all_cluster[c]["DATA"][self.c_colname[i] + "*" + x] = self.all_cluster[c]["DATA"][self.c_colname[i] + "^2"] * self.cols[x]
                
                self.all_cluster[c]["SUM_" + self.c_colname[i] + "*" + x] = np.sum(self.all_cluster[c]["DATA"][self.c_colname[i] + "*" + x])
                
                self.all_cluster[c]["SUM_" + self.c_colname[i] + "*" + x + "_DIV_" + "SUM_" + self.c_colname[i] + "^2"] = self.all_cluster[c]["SUM_" + self.c_colname[i] + "*" + x] / self.all_cluster[c]["SUM_" + self.c_colname[i] + "^2"]
                
                div[i].append(self.all_cluster[c]["SUM_" + self.c_colname[i] + "*" + x + "_DIV_" + "SUM_" + self.c_colname[i] + "^2"])

        self.center_cluster  = pd.DataFrame(np.array(div))
        self.center_cluster.columns = self.x_colname
        self.center_cluster.index = self.cluster_name
        
    def show_center_cluster(self, show_all=False):
        pd.set_option('max_rows', None if show_all else 5)

        print(self.center_cluster.head(187))

    def start_step_2(self):
        self.cluster = {}
        for i, c in enumerate(self.cluster_name):
            self.cluster[c] = pd.DataFrame()
            for x in self.x_colname:
                self.cluster[c][ "(" + x + "-" + self.v_colname[i] + ")^2"] = np.power(self.cols[x] - self.center_cluster["X1"][i], 2)
            self.cluster[c]["SUM"] = self.cluster[c].sum(axis=1)

    def calc_obj_function(self):
        self.obj_function = np.power(self.c_df, 2)
        self.obj_function.columns = [c + "^2" for c in self.c_colname]

        for i, l in enumerate(self.l_colname):
            self.obj_function[l] = np.abs(self.cluster[self.cluster_name[i]]["SUM"]) * self.obj_function[self.obj_function.columns[i]]

        self.obj_function["TOTAL"] = self.obj_function[self.l_colname].sum(axis=1)

    def show_obj_function(self, show_all=False):
        pd.set_option('max_rows', None if show_all else 5)

        print(self.obj_function.head(187))

    def calc_error(self, verbose=False):
        CUR_TOTAL = np.sum(self.obj_function["TOTAL"])
        LAST_ERROR = np.abs(CUR_TOTAL-self.PREV_TOTAL)

        if verbose:
            print()
            print("CUR_TOTAL = ",CUR_TOTAL)
            print("LAST_ERROR = ", LAST_ERROR)

        self.PREV_TOTAL = CUR_TOTAL

        return LAST_ERROR

    def calc_matrix_partition(self):
        self.matrix_partition = pd.DataFrame()
        for i, c in enumerate(self.cluster_name):
            self.matrix_partition[self.l_colname[i]] = np.power(self.cluster[c][self.cluster[c].columns[0:len(self.x_colname)]].sum(axis=1), -1)
        self.matrix_partition["LT"] = self.matrix_partition.sum(axis=1)

    def show_matrix_partition(self, show_all=False):
        pd.set_option('max_rows', None if show_all else 5)

        print(self.matrix_partition.head(187))

    def update_c(self):
        for i, c in enumerate(self.c_colname):
            self.c_df[c] = self.matrix_partition[self.l_colname[i]]/self.matrix_partition["LT"]

    def start_cluster(self, verbose=False):
        for i in range(self.MAX_ITERATION):
            self.start_step_1()
            self.start_step_2()
            self.calc_obj_function()
            error = self.calc_error(verbose)
            self.calc_matrix_partition()
            self.update_c()
            if error < self.MIN_ERROR:
                break


    

    

    