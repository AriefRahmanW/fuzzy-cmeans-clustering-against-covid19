from libraries.FuzzyCmeans import FuzzyCmeans

fcm = FuzzyCmeans(5)

# fcm.select_data("country_wise_latest.csv", "Country/Region", ["New cases", "New deaths", "New recovered"])
fcm.select_data("country_wise_latest.csv", "Country/Region", ["Deaths", "Recovered", "Active"])
# fcm.show_data()
# fcm.show_random()
# fcm.show_result(show_all=True)
fcm.start_cluster(verbose=True)
# fcm.show_center_cluster()
# fcm.show_obj_function(show_all=False)
# fcm.show_matrix_partition()

fcm.show_result(show_all=True)