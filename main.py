from libraries.FuzzyCmeans import FuzzyCmeans

from libraries.FCM import FCM

# fcm = FuzzyCmeans(5, 0.0001, 100)

# fcm.select_data("country_wise_latest.csv", "Country/Region", [   "New deaths", "New recovered"])
# fcm.select_data("segmentation_data.csv", "ID", ["Income"])
# fcm.select_data("country_wise_latest.csv", "Country/Region", ["Deaths", "Recovered", "Active"])
# fcm.show_data(show_all=True) # show selected data
# fcm.show_random() # show random_generated
# fcm.show_result(show_all=True)
# fcm.start_cluster(verbose=True)
# fcm.show_center_cluster()
# fcm.show_obj_function(show_all=False)
# fcm.show_matrix_partition()
# fcm.show_random() # show random_generated
# fcm.show_result(show_all=False)

# fcm.find_all_cluster("CLUSTER_2")

# fcm.scatter_plot("New cases", "New deaths", "New recovered")

fcm = FCM(5, 0.001, 200)

fcm.read_csv("country_wise_latest.csv", "Country/Region", ["Deaths", "Recovered", "Active"])
fcm.generate_random()
# fcm.show_result(show_all=True, group_by=True)

fcm.start(verbose=True)

fcm.show_result(show_all=True, group_by=True)

# fcm.plot()