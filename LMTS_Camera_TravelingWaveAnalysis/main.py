from Traveling_Wave_Analysis import *
from Plot_Data import *
import threading
import os
import glob

folder_path = os.getcwd() + "/Data/S17_27-03-2023/"

movie_file_list = glob.glob(folder_path + "HSCamera_movies/*.avi")
csv_rotor_file_list = glob.glob(folder_path + "/*_rot.csv")

########################################################################################################################
# Multithreaded movie analysis
# Comment that block if all movies have already been analyzed
########################################################################################################################
thread_number = 1

# Remove the already analyzed movies from the movie list
for csv_rotor_file in csv_rotor_file_list:
    for movie in movie_file_list:
        csv_movie_file = folder_path + movie.split("/")[-1].replace(".avi", "_rot.csv")
        if csv_movie_file == csv_rotor_file:
            movie_file_list.remove(movie)

all_data_mode = True
save_data = True
t = []
for th_idx in range(thread_number):
    movie_idx = int(len(movie_file_list)/thread_number)

    # Analyze the movies
    t.append(threading.Thread(target=traveling_wave_video_analysis,
                              args=(movie_file_list[movie_idx*th_idx:movie_idx*(th_idx + 1)],
                                    folder_path, all_data_mode, save_data)))
    t[-1].start()

for th_idx in range(thread_number):
    t[th_idx].join()
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Data analysis
########################################################################################################################
id_string = "PMMA-3mm"  # Use this string to sub-select which data to analyse and graph
# The experiment parameters:
camera_frame_rate = 4000  # in frame/s
actuation_time = 1  # in seconds
PMMA_2mm = 7.51  # in grams
PMMA_3mm = 10.14
PMMA_4mm = 12.17
PMMA_5mm = 17.60
PMMA_6mm = 19.00
rotor_radius = 0.03  # in meters
actuation_freq = 70  # in Hz
exp_param = (camera_frame_rate, actuation_time, PMMA_2mm, PMMA_3mm, PMMA_4mm, PMMA_5mm, PMMA_6mm,
             rotor_radius, actuation_freq)

csv_rotor_file_list = glob.glob(folder_path + "/*_" + id_string + "_*_rot.csv")
# Comment this line if you don't want to analyze the csv files anymore
traveling_wave_data_analysis(folder_path, csv_rotor_file_list, exp_param, id_string)
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Graph data here. You can choose the PMMA plate condition, and to graph soft elastomer data or PMMA plate data.
########################################################################################################################
id_string = "PMMA-3mm"  # Use this string to sub-select which data to analyse and graph ('rotor' or 'stator')
graph_data(folder_path, id_string, target='rotor', show=False)
########################################################################################################################
########################################################################################################################
