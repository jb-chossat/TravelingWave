import gc
import glob
import logging
import math
import os
import sys

import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter
from PIL import Image
from matplotlib import cm

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 10})

camera_frame_rate = 4000
actuation_time = 1  # in seconds
PMMA_2mm = 7.51  # in grams
PMMA_3mm = 10.14
PMMA_4mm = 12.17
PMMA_5mm = 17.60
PMMA_6mm = 19.00
rotor_radius = 0.03  # in meters

def graph_data(folder_path, id_string, target='all', show=False):
    """
    :param folder_path: where the data is
    :param target: 'all', 'rotor' or 'stator' depending on the graphs you want to generate
    :param show: True if you want to see the graphs as they are generated
    :return:
    """
    csv_all_data = folder_path + "Data_" + id_string + ".csv"  #  Only the csv Data_ files
    csv_filepath_list = glob.glob(folder_path + "*_" + id_string + "_*[!Data]*.csv")  # list of all csv files but the Data_ files

    # Here add all the relevant filepaths to be graphed to the csv_filepath_list
    temp = []
    if csv_filepath_list:
        for file in csv_filepath_list:
            if target == 'rotor':
                if file.find('_rot') != -1:
                    temp.append(file)
            elif target == 'stator':
                if file.find('_rot') == -1:
                    temp.append(file)

    csv_filepath_list = temp
    logging.info(f"{len(csv_filepath_list)} csv files to be graphed, target '{target}'")

    if target == 'rotor':
        logging.info(f"Rotor plotting")
        plot_detailed_acceleration(csv_filepath_list, show=show)
        plot_detailed_speed(csv_filepath_list, show=show)
        plot_displacement_macro(csv_filepath_list, id_string, show=show)
        plot_average_speed(csv_filepath_list, id_string, show=False)
        plot_rotor_torque_max(csv_all_data, id_string, show=show)
        plot_rotor_Pmech_efficiency(folder_path, show=show)

        plot_rotor_max_speed_mass(folder_path, show=show)
        plot_rotor_max_speed_mass_torque(folder_path, show=show)

        # These methods will only work if you have all the data files
        # plot_all_average_speeds(folder_path, show=True)
        # plot_average_displacement_macro(csv_filepath_list, id_string, show=True, number_of_experiments=5)

    elif target == 'stator':
        logging.info(f"Stator plotting")
        plot_XY_marker_trajectory_color_frames(csv_filepath_list, id_string, show=show)
        plot_XY_marker_trajectory_color_gif(csv_filepath_list, id_string, show=show)

# Plot all rotor instantaneous speed on as many graphs as needed, over a fixed duration
# Maybe implement with a point interpolation as shown here:
# https://stackoverflow.com/questions/23419193/second-order-gradient-in-numpy
def plot_detailed_acceleration(csv_filepath_list, show=False):
    for file in tqdm(csv_filepath_list, desc="Plotting detailed acceleration"):
        df = pd.read_csv(file, encoding='utf-8')

        filename_parts = file.replace(os.getcwd(), '').split('_')
        rotor_str = filename_parts[2]
        duration_str = filename_parts[3]
        freq_str = filename_parts[4]
        phase_diff_str = filename_parts[5]
        voltage_str = filename_parts[6]
        exp_number = filename_parts[8]


        file_path = file.strip(file.split('/')[-1]) + "RotorAcc/"
        if os.path.exists(file_path):
            pass
        else:
            os.mkdir(file_path)
        save_path = file_path + "RotorAcc_" + rotor_str + '_' + duration_str + '_' + freq_str + '_' \
                    + phase_diff_str + '_' + voltage_str + '_' + exp_number + '.jpeg'


        fig = plt.figure(1)
        x = df['Time (s)']
        y = df['Delta_x (mm)']
        acc = compute_acceleration(x, y)

        temp_str = (file.split('/')[-1]).split('_')[1]
        plt.plot(x, acc, label=temp_str)
        plt.legend()
        plt.axhline(y=0, color='red', linestyle='--')

        plt.xlabel("Time (s)")
        plt.ylabel(r"Rotor acceleration (mm/$s^2$)")
        plt.title(f"{rotor_str} rotor acceleration speed at:\n"
                  f"{freq_str}, {voltage_str}, & {phase_diff_str} of phase difference")
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200)

        if show:
            plt.show()
        plt.close()


# Plot all rotor instantaneous speed on as many graphs as needed, over a fixed duration
def plot_detailed_speed(csv_filepath_list, show=False):
    for file in tqdm(csv_filepath_list, desc="Plotting raw speed"):
        df = pd.read_csv(file, encoding='utf-8')

        filename_parts = file.replace(os.getcwd(), '').split('_')
        rotor_str = filename_parts[2]
        duration_str = filename_parts[3]
        freq_str = filename_parts[4]
        phase_diff_str = filename_parts[5]
        voltage_str = filename_parts[6]
        exp_number = filename_parts[8]


        file_path = file.strip(file.split('/')[-1]) + "RotorSpeed/"
        if os.path.exists(file_path):
            pass
        else:
            os.mkdir(file_path)
        save_path = file_path + "RotorsInstSpeed_" + rotor_str + '_' + duration_str + '_' + freq_str + '_' \
                    + phase_diff_str + '_' + voltage_str + '_' + exp_number + '.jpeg'

        plt.figure(1)
        x = df['Time (s)']
        y = df['Delta_x (mm)']
        v = compute_velocity(y) * 1000  # Convert to mm/s

        temp_str = (file.split('/')[-1]).split('_')[1]
        plt.plot(x,v, label=temp_str)

        plt.legend()
        plt.axhline(y=0, color='red', linestyle='--')

        plt.xlabel("Time (s)")
        plt.ylabel("Rotor speed (mm/s)")
        plt.title(f"{rotor_str} rotor speed at: \n{freq_str}, {voltage_str}, & {phase_diff_str} of phase difference")
        plt.tight_layout()

        plt.savefig(save_path, dpi=1200)

        if show:
            plt.show()
        plt.close()


# Plot all rotor displacement on a single graph, over the whole test duration
def plot_displacement_macro(csv_filepath_list, id_string, show=False):

    cmap = plt.cm.tab20c
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cmap(np.linspace(0, 1, 19)))
    cm = 1 / 2.54  # used to convert in centimeters
    custom_legend = []

    for file in csv_filepath_list:
        if id_string in file:
            df = pd.read_csv(file, encoding='utf-8')

            # Data
            x = df['Time (s)']
            y = df['Delta_x (mm)']
            d = compute_total_displacement(y)
            r_deg = 360 * d / (2 * math.pi * 30)

            # Fit lines
            a, b = np.polyfit(x, d, 1)
            avg_rpm = (a * 60)/(2*math.pi*30)

            temp_str = (file.split('/')[-1]).split('_')[4]
            custom_legend.append(temp_str)
            # + ', average speed:{:.2f}'.format(float(a)) \
            #            + ' mm/s (RPM:{:.2f})'.format(float(avg_rpm)))
            if csv_filepath_list.index(file) % 3 == 0:
                plt.plot(x, r_deg, marker='*', markevery=200, label=temp_str)
            elif csv_filepath_list.index(file) % 3 == 1:
                plt.plot(x, r_deg, marker='+', markevery=200, label=temp_str)
            else:
                plt.plot(x, r_deg, marker='x', markevery=200, label=temp_str)

    fig = plt.figure(1, figsize=(19*cm, 12*cm))
    ax = plt.subplot(111)
    plt.xlabel("Time (s)")
    plt.ylabel("Rotor cumulative rotation (deg)")
    plt.title(id_string)

    ax.legend(sorted(custom_legend))
    plt.tight_layout()
    file_path = csv_filepath_list[0].strip(csv_filepath_list[0].split('/')[-1])
    filename_parts = (csv_filepath_list[0].split('/')[-1]).split('_')

    plt.savefig(
        file_path + filename_parts[0] + '_RotorCumulativeDisplacement_' + id_string + '_' + filename_parts[2] \
        + '_' + filename_parts[3] + '_' + filename_parts[4] + '_' + filename_parts[5] + '.jpeg', dpi=1200)
    if show:
        plt.show()
    plt.close()


def plot_average_displacement_macro(csv_filepath_list, id_string, show=False, number_of_experiments=5):

    cmap = plt.cm.tab20c
    cm = 1 / 2.54  # used to convert in centimeters

    displacement_list = []
    rotation = []
    phase_diff = []
    # average_displacement = []
    # std_average_displacement = []

    for file in csv_filepath_list:
        df = pd.read_csv(file, encoding='utf-8')

        x = df[df['Time (s)'] < 1.0]['Time (s)']
        y = df[df['Time (s)'] < 1.0]['Delta_x (mm)']

        y_deg = (360*y) / (2*math.pi*rotor_radius*1000)
        displacement_list.append(compute_total_displacement(y_deg))
        phase_diff.append(float((file.split('/')[-1]).split('_')[4].replace('deg', '')))

    sorted_phase_diff = sorted(phase_diff)
    sorted_phase_diff_selected = [sorted_phase_diff[0],
                                  sorted_phase_diff[9*number_of_experiments],
                                  sorted_phase_diff[18*number_of_experiments],
                                  sorted_phase_diff[27*number_of_experiments],
                                  sorted_phase_diff[4*number_of_experiments],
                                  sorted_phase_diff[22*number_of_experiments],
                                  ]

    fig = plt.figure(1, figsize=(8.8 * cm, 6 * cm)) #figsize=(19*cm, 10*cm))
    ax = plt.subplot(111)

    for phase in sorted_phase_diff_selected:

        error_bar_pos_step = 400

        idx1 = phase_diff.index(phase)
        idx2 = phase_diff.index(phase, idx1+1)
        idx3 = phase_diff.index(phase, idx2+1)
        idx4 = phase_diff.index(phase, idx3+1)
        idx5 = phase_diff.index(phase, idx4+1)

        average_displacement = (displacement_list[idx1] +
                                 displacement_list[idx2] +
                                 displacement_list[idx3] +
                                 displacement_list[idx4] +
                                 displacement_list[idx5]) / 5.0

        label_str = f"{int(phase)}\u00b0 \u03A6 difference"
        ax.plot(x, average_displacement, label=label_str)
        ax.scatter(x[0::error_bar_pos_step], average_displacement[0::error_bar_pos_step],
                   marker=".", color="k", edgecolors=None, s=25, zorder=1)# , color="k"
        std = []

        for data_idx in range(0, len(displacement_list[idx1])):
            data1 = displacement_list[idx1][data_idx]
            data2 = displacement_list[idx2][data_idx]
            data3 = displacement_list[idx3][data_idx]
            data4 = displacement_list[idx4][data_idx]
            data5 = displacement_list[idx5][data_idx]
            std.append(np.std([data1, data2, data3, data4, data5]))

        plt.errorbar(x[0::error_bar_pos_step], average_displacement[0::error_bar_pos_step],
                     yerr=std[0::error_bar_pos_step], capsize=2, ls="None", color='black',
                     elinewidth=0.5)  # fmt='-o'# errorevery=250

    plt.text(0.86, 0.92, 'Φ difference = 270°', ha='left', va='top', transform=ax.transAxes)
    plt.text(0.86, 0.77, 'Φ difference = 220°', ha='left', va='top', transform=ax.transAxes)
    plt.text(0.86, 0.60, 'Φ difference = 180°', ha='left', va='top', transform=ax.transAxes)
    plt.text(0.86, 0.54, 'Φ difference = 0°', ha='left', va='top', transform=ax.transAxes)
    plt.text(0.86, 0.35, 'Φ difference = 40°', ha='left', va='top', transform=ax.transAxes)
    plt.text(0.86, 0.16, 'Φ difference = 90°', ha='left', va='top', transform=ax.transAxes)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel("Time (s)")
    plt.ylabel("Rotor cumulative rotation (°)")
    plt.ylim(-25, 25)
    plt.xlim(0, 1.2)
    plt.tight_layout()

    fig.subplots_adjust(bottom=0.185, left=0.16, right=0.76, top=0.97, hspace=0.1)

    file_path = csv_filepath_list[0].strip(csv_filepath_list[0].split('/')[-1])
    filename_parts = (csv_filepath_list[0].split('/')[-1]).split('_')

    save_path = file_path + filename_parts[0] + '_RotorCumulativeDisplacement_' + id_string + '_' + filename_parts[2] \
        + '_' + filename_parts[3] + '_' + filename_parts[4] + '_' + filename_parts[5]
    plt.savefig(save_path + '.png', dpi=600)
    plt.savefig(save_path + '.eps', dpi=600, format='eps')
    if show:
        plt.show()
    plt.close()


def plot_average_speed(csv_filepath_list, id_string, show=False):
    avg_speed_values = []
    phase_diff = []

    for file in csv_filepath_list:
        if id_string in file:
            df = pd.read_csv(file, encoding='utf-8')

            # Data
            x = df['Time (s)']
            y = df['Delta_x (mm)']
            d = compute_total_displacement(y)

            # Fit lines
            a, b = np.polyfit(x, d, 1)
            avg_speed_values.append(a)
            phase_diff.append(float((file.split('/')[-1]).split('_')[4].replace('deg', '')))

    sorted_avg_speed_values = [x for _, x in sorted(zip(phase_diff, avg_speed_values))]
    sorted_avg_speed_values_rpm = np.array(sorted_avg_speed_values) * 60 / ( 2 * math.pi * 30)
    sorted_phase_diff = sorted(phase_diff)

    std_sorted_avg_speed_values_rpm = []
    averaged_sorted_avg_speed_values_rpm = []
    decimated_sorted_phase_diff = sorted_phase_diff[::5]

    # Average all sequences data and find the standard deviation
    for value_idx in range(0, len(sorted_phase_diff), 5):
        averaged_sorted_avg_speed_values_rpm.append((sorted_avg_speed_values_rpm[value_idx] +
                                                sorted_avg_speed_values_rpm[value_idx+1] +
                                                sorted_avg_speed_values_rpm[value_idx+2] +
                                                sorted_avg_speed_values_rpm[value_idx+3] +
                                                sorted_avg_speed_values_rpm[value_idx+4]) / 5)
        std_sorted_avg_speed_values_rpm.append(np.std([sorted_avg_speed_values_rpm[value_idx],
                                                sorted_avg_speed_values_rpm[value_idx+1],
                                                sorted_avg_speed_values_rpm[value_idx+2],
                                                sorted_avg_speed_values_rpm[value_idx+3],
                                                sorted_avg_speed_values_rpm[value_idx+4]]))
    # print(len(std_sorted_avg_speed_values_rpm))
    # print(std_sorted_avg_speed_values_rpm)

    fig = plt.figure(1)  # figsize=(8.25,5)
    ax = plt.subplot(111)
    plt.xlabel("Phase difference (deg)")
    plt.ylabel("Rotor average rotation speed (rpm)")
    plt.title(id_string)
    plt.axhline(y=0, color='grey', linestyle='--')
    plt.xticks(rotation=60)
    plt.errorbar(decimated_sorted_phase_diff, averaged_sorted_avg_speed_values_rpm,
             yerr=std_sorted_avg_speed_values_rpm, ls = "None", color='black', capsize=2) # fmt='-o'
    plt.scatter(decimated_sorted_phase_diff, averaged_sorted_avg_speed_values_rpm, marker=".", color="black")
    plt.plot(decimated_sorted_phase_diff, averaged_sorted_avg_speed_values_rpm, color="blue")

    plt.tight_layout()

    file_path = csv_filepath_list[0].strip(csv_filepath_list[0].split('/')[-1])
    filename_parts = (csv_filepath_list[0].split('/')[-1]).split('_')
    plt.savefig(
        file_path + filename_parts[0] + '_RotorAverageRotationSpeed_' + id_string + '_' + filename_parts[2] \
        + '_' + filename_parts[3] + '_' + filename_parts[4] + '_' + filename_parts[5] + '.jpeg', dpi=1200)

    if show:
        plt.show()
    plt.close()


def plot_all_average_speeds(folder_path, show=True):
    # Used to map the rotor's average rotation speed for various conditions on the same graph
    # sub_string = ['15Hz', '20Hz', '25Hz', '30Hz', '35Hz', '40Hz']

    # csv_filepath_list = glob.glob(folder_path + "/*_rot.csv")
    # sub_string = ['PMMA-2mm', 'PMMA-3mm', 'PMMA-4mm', 'PMMA-5mm', 'PMMA-6mm']
    ylabels = ["2mm rotor", "3mm rotor", "4mm rotor", "5mm rotor", "6mm rotor"]

    csv_filepath_list_2mm = glob.glob(folder_path + "/*_PMMA-2mm_*_rot.csv")
    csv_filepath_list_3mm = glob.glob(folder_path + "/*_PMMA-3mm_*_rot.csv")
    csv_filepath_list_4mm = glob.glob(folder_path + "/*_PMMA-4mm_*_rot.csv")
    csv_filepath_list_5mm = glob.glob(folder_path + "/*_PMMA-5mm_*_rot.csv")
    csv_filepath_list_6mm = glob.glob(folder_path + "/*_PMMA-6mm_*_rot.csv")
    csv_filepath_list = [csv_filepath_list_2mm, csv_filepath_list_3mm, csv_filepath_list_4mm, csv_filepath_list_5mm,
                         csv_filepath_list_6mm]

    cm = 1 / 2.54  # used to convert in centimeters
    fig, axes = plt.subplots(nrows=len(csv_filepath_list), ncols=1, sharex='all', sharey='all', figsize=(8.8*cm, 20*cm))

    for file_array in tqdm(csv_filepath_list, "Generating graphs"):
        avg_speed_values = []
        phase_diff = []

        for file in file_array:
            df = pd.read_csv(file, encoding='utf-8')

            # Data
            x = df['Time (s)']
            y = df['Delta_x (mm)']
            d = compute_total_displacement(y)

            # Fit lines
            a, b = np.polyfit(x, d, 1)
            avg_speed_values.append(a)
            phase_diff.append(float((file.split('/')[-1]).split('_')[4].replace('deg', '')))

        sorted_avg_speed_values = [x for _, x in sorted(zip(phase_diff, avg_speed_values))]
        sorted_avg_speed_values_rpm = np.array(sorted_avg_speed_values) * 60 / (2 * math.pi * 30)
        sorted_phase_diff = sorted(phase_diff)

        decimated_sorted_phase_diff = sorted_phase_diff[::5]
        averaged_sorted_avg_speed_values_rpm = []
        std_sorted_avg_speed_values_rpm = []

        # for value_idx in range(0, len(sorted_phase_diff), 5):
        index_phase_diff = -90
        for value_idx in range(0, len(sorted_phase_diff), 5):
            averaged_sorted_avg_speed_values_rpm.append((sorted_avg_speed_values_rpm[value_idx + index_phase_diff] +
                                                         sorted_avg_speed_values_rpm[value_idx + 1 + index_phase_diff] +
                                                         sorted_avg_speed_values_rpm[value_idx + 2 + index_phase_diff] +
                                                         sorted_avg_speed_values_rpm[value_idx + 3 + index_phase_diff] +
                                                         sorted_avg_speed_values_rpm[value_idx + 4 + index_phase_diff])
                                                        / 5)
            std_sorted_avg_speed_values_rpm.append(np.std([sorted_avg_speed_values_rpm[value_idx + index_phase_diff],
                                                           sorted_avg_speed_values_rpm[value_idx + 1 + index_phase_diff],
                                                           sorted_avg_speed_values_rpm[value_idx + 2 + index_phase_diff],
                                                           sorted_avg_speed_values_rpm[value_idx + 3 + index_phase_diff],
                                                           sorted_avg_speed_values_rpm[value_idx + 4 + index_phase_diff]]))

        legend_string = "m" + str((csv_filepath_list.index(file_array) + 1))
        scatter_handle = axes[csv_filepath_list.index(file_array)].scatter(decimated_sorted_phase_diff,
                                                          averaged_sorted_avg_speed_values_rpm, marker=".",
                                                          color="red", label=legend_string)
                                                          # label=ylabels[csv_filepath_list.index(file_array)])

        axes[csv_filepath_list.index(file_array)].errorbar(decimated_sorted_phase_diff,
                                                           averaged_sorted_avg_speed_values_rpm,
                                                           yerr=std_sorted_avg_speed_values_rpm,
                                                           ls="None", color='red', capsize=2)  # fmt='-o'

        line_handle = axes[csv_filepath_list.index(file_array)].plot(decimated_sorted_phase_diff,
                                                       averaged_sorted_avg_speed_values_rpm,
                                                       color="black")

        axes[csv_filepath_list.index(file_array)].legend(frameon=False, loc='lower right')
        axes[csv_filepath_list.index(file_array)].set_xlim(0, 350)
        axes[csv_filepath_list.index(file_array)].set_ylim(-5.5, 5.5)
        axes[csv_filepath_list.index(file_array)].axhline(y=0, color='grey', linestyle='dotted')

        plt.xlabel(r"$\theta_{diff}~(^\circ)$")
        plt.xticks(np.arange(0, 360, step=50))  # Set label locations.
        axes[2].set_ylabel("Average rotation velocity (RPM)")

    fig.subplots_adjust(bottom=0.06, left=0.125, right=0.97, top=0.995, hspace=0.1)
    plt.savefig(folder_path + 'RotorAverageRotationSpeed_AllData.eps', dpi=600, format='eps')
    plt.savefig(folder_path + 'RotorAverageRotationSpeed_AllData.png', dpi=600, format='png')

    if show:
        plt.show()
    plt.close()


def plot_XY_marker_trajectory_color_frames(csv_filepath_list, id_string, show=False):

    f_start = 1985
    f_1 = 2000
    f_2 = 2014
    f_3 = 2028
    f_end = 2043

    # f_4 = 2043

    for file in tqdm(csv_filepath_list, desc="Plotting marker (colored) trajectory"):

        filename_parts = file.replace(os.getcwd(), '').split('_')
        rotor_str = filename_parts[2]
        duration_str = filename_parts[3]
        freq_str = filename_parts[4]
        phase_diff_str = filename_parts[5]
        voltage_str = filename_parts[6]
        exp_number = filename_parts[8]

        df = pd.read_csv(file, encoding='utf-8')
        all_data_file = file.replace(file.split('/')[-1], '') + "Data_" + id_string + ".csv"
        all_data_df = pd.read_csv(all_data_file, encoding='utf-8')

        width_mm = 41.5
        height_mm = 32
        width_inches = width_mm / 25.4
        height_inches = height_mm / 25.4
        fig = plt.figure(figsize=(width_inches, height_inches))
        ax = plt.gca()

        for point in range(f_start, f_end):
            x = df['blob_x (mm)'][point]
            y = df['blob_y (mm)'][point]

            if f_start <= point < f_1:
                alpha = (1 + point - f_start) / (f_1 - f_start)
                ax.plot(x, y, marker='o', markersize=2, alpha=alpha, color='blue')
            elif f_1 <= point < f_2:
                alpha = (1 + point - f_1) / (f_2 - f_1)
                ax.plot(x, y, marker='o', markersize=2, alpha=alpha, color='green')
            elif f_2 <= point < f_3:
                alpha = (1 + point - f_2) / (f_3 - f_2)
                ax.plot(x, y, marker='o', markersize=2, alpha=alpha, color='red')
            elif f_3 <= point < f_end:
                alpha = (1 + point - f_3) / (f_end - f_3)
                ax.plot(x, y, marker='o', markersize=2, alpha=alpha, color='orange')
            else:
                ax.plot(x, y, marker='o', markersize=2, alpha=1, color='grey')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.set_xticks([-0.06, 0.06])
        ax.set_yticks([-0.6, 0.6])

        plt.xlabel("X position (mm)", loc='right', labelpad=-66)
        plt.ylabel("Y position (mm)", loc='top', labelpad=30)

        file_path = file.strip(file.split('/')[-1]) + "MarkerTrajectory/"
        if os.path.exists(file_path):
            pass
        else:
            os.mkdir(file_path)
        save_path = file_path + "Partial_trajectories/MarkerTrajectoryColor_" + rotor_str + '_' + duration_str + '_' \
                    + freq_str + '_' + phase_diff_str + '_' + voltage_str + '_' + exp_number + '_' + str(f_start) + \
                    '_' + str(f_end) + '.jpeg'
        print("Saving: " + save_path)

        plt.xlim(-0.065, 0.065)
        plt.ylim(-0.65, 0.65)

        # Manually set the spacing around the figure
        plt.subplots_adjust(top=0.88, bottom=0.01, right=0.95, left=0.18)

        plt.savefig(save_path, dpi=600)
        # plt.savefig(save_path, dpi=fig.dpi)

        if show:
            plt.show()
        plt.close()


def plot_XY_marker_trajectory_color_gif(csv_filepath_list, id_string, show=False):
    for file in tqdm(csv_filepath_list, desc="Plotting marker (colored) trajectory", position=0, leave=True):

        filename_parts = file.replace(os.getcwd(), '').split('_')
        rotor_str = filename_parts[2]
        duration_str = filename_parts[3]
        freq_str = filename_parts[4]
        phase_diff_str = filename_parts[5]
        voltage_str = filename_parts[6]
        exp_number = filename_parts[8]

        # Reads the specific target file
        df = pd.read_csv(file, encoding='utf-8')

        # Reads the table with all the files' results
        all_data_file = file.replace(file.split('/')[-1], '') + "Data_" + id_string + ".csv"
        all_data_df = pd.read_csv(all_data_file, encoding='utf-8')

        # Create a list to store frames
        frames = []
        file_path = os.getcwd() + '/Data/S17_27-03-2023/MarkerTrajectory/GIF/'
        start = 10
        stop = 4000
        step = 2
        movie_speed = 0.05
        gif_delay = (1.0 * step) / (camera_frame_rate * movie_speed)
        # arrow_max_length = 0.1
        # dx = 0.0

        file_name = file_path + "MarkerTrajectoryColor_" + rotor_str + '_' + duration_str + '_' + freq_str + '_' \
                    + phase_diff_str + '_' + voltage_str + '_' + exp_number + '.gif'

        # Add a red arrow indicating rotor rotation direction, which length is proportional to the rotation speed
        # for idx in range(len(all_data_df["Filename"])):
        #     if file.replace('.csv', '_rot.csv') == all_data_df.loc[idx]["Filename"]:  # and \
        #         # Counterclockwise
        #         if all_data_df.loc[idx]["Avg velocity (mm/s)"] > 0.0:
        #             dx = arrow_max_length * abs(all_data_df.loc[idx]["Avg velocity (mm/s)"]) / \
        #                  max(abs(all_data_df["Avg velocity (mm/s)"]))
        #         # Clockwise
        #         elif all_data_df.loc[idx]["Avg velocity (mm/s)"] < 0.0:
        #             dx = -arrow_max_length * abs(all_data_df.loc[idx]["Avg velocity (mm/s)"]) / \
        #                  max(abs(all_data_df["Avg velocity (mm/s)"]))

        for point in tqdm(range(start, stop, step), desc="Generating individual point plots", position=0, leave=True):
            fig = plt.figure()
            plt.title(f'Time {point/camera_frame_rate}s \nFrame {point}')
            plt.xlim(-0.06, 0.06)
            plt.ylim(-0.6, 0.6)
            ax = plt.gca()

            # plt.arrow(0, 0.75, dx, 0, color='black', width=0.05, head_length=0.005)
            ax.plot(df['blob_x (mm)'][point], df['blob_y (mm)'][point], marker='o', markersize=1, color='red', alpha=1)
            ax.plot(df['blob_x (mm)'][point-1], df['blob_y (mm)'][point-1], marker='o', markersize=1, color='red', alpha=0.9)
            ax.plot(df['blob_x (mm)'][point-2], df['blob_y (mm)'][point-2], marker='o', markersize=1, color='red', alpha=0.8)
            ax.plot(df['blob_x (mm)'][point-3], df['blob_y (mm)'][point-3], marker='o', markersize=1, color='red', alpha=0.7)
            ax.plot(df['blob_x (mm)'][point-4], df['blob_y (mm)'][point-4], marker='o', markersize=1, color='red', alpha=0.6)
            ax.plot(df['blob_x (mm)'][point-5], df['blob_y (mm)'][point-5], marker='o', markersize=1, color='red', alpha=0.5)
            ax.plot(df['blob_x (mm)'][point-6], df['blob_y (mm)'][point-6], marker='o', markersize=1, color='red', alpha=0.4)
            ax.plot(df['blob_x (mm)'][point-7], df['blob_y (mm)'][point-7], marker='o', markersize=1, color='red', alpha=0.3)
            ax.plot(df['blob_x (mm)'][point-8], df['blob_y (mm)'][point-8], marker='o', markersize=1, color='red', alpha=0.2)
            ax.plot(df['blob_x (mm)'][point-9], df['blob_y (mm)'][point-9], marker='o', markersize=1, color='red', alpha=0.1)

            plt.savefig(file_path + f'frame_{point}.png', dpi=120)
            frames.append(Image.open(file_path + f'frame_{point}.png'))

            # Close the plot to free up resources
            plt.close('all')

        frames[0].save(file_name, save_all=True, append_images=frames[1:], duration=gif_delay, loop=0)

        # Cleanup: Remove the temporary image files
        for point in range(start, stop, step):
            os.remove(file_path + f'frame_{point}.png')

        del df, frames
        plt.close('all')
        gc.collect()

def plot_rotor_torque_max(csv_all_data_file, id_string, show=False):
    # Do the torque average only on the envelope from 0s to 1s
    # torque_start = int(0.05 * camera_frame_rate)
    torque_end = camera_frame_rate*1

    for nb_exp in range(1, 6, 1):
        # Load the data from the "all data csv" file
        df_all_data = pd.read_csv(csv_all_data_file, encoding='utf-8')
        # Find the filepath for the clockwise and counter-clockwise max velocity cases
        csv_max_velocity_ccw = df_all_data[(df_all_data["Sequence #"] == nb_exp) &
                                           (df_all_data["Average max torque (mN.m)"] < 0.0)]["Filename"].iloc[0]
        csv_max_velocity_cw = df_all_data[(df_all_data["Sequence #"] == nb_exp) &
                                          (df_all_data["Average max torque (mN.m)"] > 0.0)]["Filename"].iloc[0]

        # Load these particular csv files
        df_ccw = pd.read_csv(csv_max_velocity_ccw, encoding='utf-8')
        df_cw = pd.read_csv(csv_max_velocity_cw, encoding='utf-8')

        filename_parts_ccw = csv_max_velocity_ccw.split('_')
        rotor_str = filename_parts_ccw[3]
        duration_str = filename_parts_ccw[4]
        freq_str = filename_parts_ccw[5]
        phase_diff_str = filename_parts_ccw[6]
        voltage_str = filename_parts_ccw[7]
        exp_number = filename_parts_ccw[9]

        # Saving figures paths
        file_path_ccw = csv_all_data_file.strip(csv_all_data_file.split('/')[-1]) + "Torque/Ccw_torqueMax" + id_string \
                        + '_' + exp_number +'_' + phase_diff_str + ".jpeg"

        # Data for the ccw torque series
        x = df_ccw['Time (s)']
        y = df_ccw['Delta_x (mm)']
        mean_torque = df_all_data[(df_all_data["Sequence #"] == nb_exp) & \
                                  (df_all_data["Average max torque (mN.m)"] < 0.0)]["Average max torque (mN.m)"].iloc[0]
        torque_ccw = compute_torque(csv_max_velocity_ccw, x, y)
        envelope_min, envelope_max = hl_envelopes_idx(torque_ccw[:torque_end], dmin=25, dmax=25)

        fig1 = plt.figure(1)
        plt.plot(x, torque_ccw)

        # Here do envelope and take the average value
        plt.plot(x[envelope_min], torque_ccw[envelope_min], label='Data envelope')
        plt.hlines(mean_torque, 0, 1, colors='red', label='Mean torque')
        plt.ylim(-4, 4)
        plt.xlim(0, 1)

        plt.legend()
        plt.axhline(y=0, color='black', linestyle='--')
        # plt.title(f"{rotor_str} rotor torque at: \n{freq_str}, {voltage_str}, & {phase_diff_str} of phase difference")
        if int(phase_diff_str.replace("deg","")) > 180:
            direction = "clockwise rotation"
        else:
            direction = "counter-clockwise rotation"

        if id_string == "PMMA-5mm":
            rotor = "m4"
        else:
            rotor = id_string

        plt.title(f"Rotor {rotor} torque data: Φ={phase_diff_str}, {direction}")

        plt.xlabel("Time (s)")
        plt.ylabel("Motor torque (mN.m)")
        plt.tight_layout()
        plt.savefig(file_path_ccw, dpi=600)

        if show:
            plt.show()
        plt.close()

        filename_parts_cw = csv_max_velocity_cw.split('_')
        rotor_str = filename_parts_cw[3]
        duration_str = filename_parts_cw[4]
        freq_str = filename_parts_cw[5]
        phase_diff_str = filename_parts_cw[6]
        voltage_str = filename_parts_cw[7]
        exp_number = filename_parts_cw[9]

        # Saving figures paths
        file_path_cw = csv_all_data_file.strip(csv_all_data_file.split('/')[-1]) + "Torque/Cw_torqueMax" + id_string \
                        + '_' + exp_number + '_' + phase_diff_str + ".jpeg"

        # Data for the cw torque series
        x = df_cw['Time (s)']
        y = df_cw['Delta_x (mm)']
        mean_torque = df_all_data[(df_all_data["Sequence #"] == nb_exp) & \
                                  (df_all_data["Average max torque (mN.m)"] > 0.0)]["Average max torque (mN.m)"].iloc[0]
        torque_cw = compute_torque(csv_max_velocity_cw, x, y)
        envelope_min, envelope_max = hl_envelopes_idx(torque_cw[:torque_end], dmin=25, dmax=25)

        fig2 = plt.figure(2)
        plt.plot(x, torque_cw)
                 # markevery=markers_idx, marker='o', markersize=4, markerfacecolor='tab:red')

        # Here do envelope and take the average value
        plt.plot(x[envelope_max], torque_cw[envelope_max], label='Data envelope')
        plt.hlines(mean_torque, 0, 1, colors='red', label='Mean torque')
        plt.ylim(-4, 4)
        plt.xlim(0, 1)

        # plt.title(f"{rotor_str} rotor torque at: \n{freq_str}, {voltage_str}, & {phase_diff_str} of phase difference")
        if int(phase_diff_str.replace("deg","")) > 180:
            direction = "clockwise rotation"
        else:
            direction = "counter-clockwise rotation"

        if id_string == "PMMA-5mm":
            rotor = "m4"
        else:
            rotor = id_string

        plt.title(f"Rotor {rotor} torque data: Φ={phase_diff_str}, {direction}")
        plt.legend()
        plt.axhline(y=0, color='black', linestyle='--')

        plt.xlabel("Time (s)")
        plt.ylabel("Motor torque (mN.m)")
        plt.tight_layout()
        plt.savefig(file_path_cw, dpi=600)

        if show:
            plt.show()
        plt.close()


def plot_rotor_Pmech_efficiency(folder_path, show=False):
    data_files = [folder_path + "Data_PMMA-2mm.csv", folder_path + "Data_PMMA-3mm.csv",
                  folder_path + "Data_PMMA-4mm.csv", folder_path + "Data_PMMA-5mm.csv",
                  folder_path + "Data_PMMA-6mm.csv"]

    avg_pmech = []
    max_pmech = 0.0
    avg_pmech_cw = []
    avg_pmech_ccw = []
    avg_pmech_std = []
    avg_pmech_cw_std = []
    avg_pmech_ccw_std = []

    r_coils = 5.51
    v_pp = 10
    v_rms = v_pp / (2 * math.sqrt(2))
    # This is only for one op amp, there are 2 op amps, so I need to multiply p_in by 2 as well
    p_in = 2 * v_rms ** 2 / r_coils

    for file in data_files:
        # Load the data from the "all data csv" file
        df = pd.read_csv(file, encoding='utf-8')

        # Average all the Pmech data that is > 0.0, and computes the standard deviation
        avg_pmech.append(np.mean(df.loc[df["Pmech (mW)"] != 0.0]["Pmech (mW)"].to_numpy()))
        avg_pmech_std.append(np.std(df.loc[df["Pmech (mW)"] != 0.0]["Pmech (mW)"].to_numpy()))

        # Max pmech
        max_pmech = max(df.loc[df["Pmech (mW)"] != 0.0]["Pmech (mW)"].to_numpy())

        # Average all Pmech data for clockwise and counterclockwise rotation
        avg_pmech_cw.append(np.mean(df.loc[(df["Pmech (mW)"] != 0.0) & (df["Avg velocity (rad/s)"] < 0.0)]
                                    ["Pmech (mW)"].to_numpy()))
        avg_pmech_ccw.append(np.mean(df.loc[(df["Pmech (mW)"] != 0.0) & (df["Avg velocity (rad/s)"] > 0.0)]
                                     ["Pmech (mW)"].to_numpy()))
        # Standard deviation
        avg_pmech_cw_std.append(np.std(df.loc[(df["Pmech (mW)"] != 0.0) & (df["Avg velocity (rad/s)"] < 0.0)]
                                       ["Pmech (mW)"].to_numpy()))
        avg_pmech_ccw_std.append(np.std(df.loc[(df["Pmech (mW)"] != 0.0) & (df["Avg velocity (rad/s)"] > 0.0)]
                                        ["Pmech (mW)"].to_numpy()))

    fig1, ax1 = plt.subplots()
    x_axis = [1, 2, 3, 4, 5]
    plt.bar(np.subtract(x_axis, 0.2), avg_pmech_ccw, width=0.4, label='CCW rotation', yerr=avg_pmech_ccw_std,
            capsize=10)
    plt.bar(np.add(x_axis, 0.2), avg_pmech_cw, width=0.4, label='CW rotation', yerr=avg_pmech_cw_std, capsize=10)
    ax1.set_xticks(x_axis, ['m1', 'm2', 'm3', 'm4', 'm5'])
    plt.legend()
    plt.xlabel("Rotor mass")
    plt.ylabel("Pmech (mW)")
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(88.9 / 25.4, 60 / 25.4))
    # fig2, ax2 = plt.subplots()
    plt.bar(x_axis, avg_pmech, yerr=avg_pmech_std, capsize=10)
    ax2.set_xticks(x_axis, ['m1', 'm2', 'm3', 'm4', 'm5'])
    plt.xlabel("Rotor mass")
    plt.ylabel("Pmech (mW)")
    pmech_ylim = 1.15
    ax2.set_ylim(0, pmech_ylim)
    ax3 = ax2.twinx()
    ax3.set_ylim(0, pmech_ylim/p_in)
    plt.ylabel("Efficiency (%)")
    plt.tight_layout()

    # Saving filepath
    file_path = folder_path + "Pmech_eff.jpeg"
    plt.savefig(file_path, dpi=1200)

    if show:
        plt.show()
    plt.close()

def plot_rotor_max_speed_mass(folder_path, show=False):
    csv_all_data_file = [folder_path + "Data_PMMA-2mm.csv", folder_path + "Data_PMMA-3mm.csv",
                         folder_path + "Data_PMMA-4mm.csv", folder_path + "Data_PMMA-5mm.csv",
                         folder_path + "Data_PMMA-6mm.csv"]

    average_PMMA_speed_ccw = []
    average_PMMA_speed_cw = []
    std_PMMA_speed_ccw = []
    std_PMMA_speed_cw = []

    PMMA_2mm = 7.51  # in grams
    PMMA_3mm = 10.14
    PMMA_4mm = 12.17
    PMMA_5mm = 17.60
    PMMA_6mm = 19.00
    rotors_mass = [PMMA_2mm, PMMA_3mm, PMMA_4mm, PMMA_5mm, PMMA_6mm]

    cm = 1 / 2.54  # used to convert in centimeters

    for file in csv_all_data_file:
        # Load the data from the "all data csv" file
        df_all_data = pd.read_csv(file, encoding='utf-8')
        PMMA_speed_ccw = df_all_data[(df_all_data["Pmech (mW)"] != 0.0) & (df_all_data["Avg velocity (rad/s)"] > 0.0)] ["Avg velocity (rpm)"]
        PMMA_speed_cw = -1*df_all_data[(df_all_data["Pmech (mW)"] != 0.0) & (df_all_data["Avg velocity (rad/s)"] < 0.0)] ["Avg velocity (rpm)"]

        average_PMMA_speed_ccw.append(np.mean(PMMA_speed_ccw))
        std_PMMA_speed_ccw.append(np.std(PMMA_speed_ccw))
        average_PMMA_speed_cw.append(np.mean(PMMA_speed_cw))
        std_PMMA_speed_cw.append(np.std(PMMA_speed_cw))

    fig, ax1 = plt.subplots(figsize=(8.8*cm, 6*cm))

    plt.plot(rotors_mass, average_PMMA_speed_ccw, '-x', label='Counterclockwise rotation')
    plt.errorbar(rotors_mass, average_PMMA_speed_ccw, yerr=std_PMMA_speed_ccw, ls = "None", color='black', capsize=2)
    plt.plot(rotors_mass, average_PMMA_speed_cw, "-+", label='Clockwise rotation')
    plt.errorbar(rotors_mass, average_PMMA_speed_cw, yerr=std_PMMA_speed_cw, ls = "None", color='black', capsize=2)

    legend = plt.legend(borderpad=0.3, borderaxespad=0.3, loc="lower left")
    frame = legend.get_frame()
    frame.set_facecolor('white')

    plt.ylim(-0.1, 6)
    plt.xlim(7.3, 19.5)
    plt.axvline(x=PMMA_2mm, linestyle="--", color="grey", linewidth=0.5)
    plt.axvline(x=PMMA_3mm, linestyle="--", color="grey", linewidth=0.5)
    plt.axvline(x=PMMA_4mm, linestyle="--", color="grey", linewidth=0.5)
    plt.axvline(x=PMMA_5mm, linestyle="--", color="grey", linewidth=0.5)
    plt.axvline(x=PMMA_6mm, linestyle="--", color="grey", linewidth=0.5)

    plt.ylabel("Peak rotation velocity (RPM)")
    plt.xlabel("PMMA rotor thickness (mm)")
    plt.xticks([PMMA_2mm, PMMA_3mm, PMMA_4mm, PMMA_5mm, PMMA_6mm],
               ['2mm', '3mm','4mm','5mm','6mm'])
    ax2 = ax1.twiny()
    plt.xlim(7.3, 19.5)
    plt.xlabel("PMMA rotor mass (g)")

    plt.tight_layout()

    # Saving filepath
    file_path = folder_path + "Velocity_mass.png"
    plt.savefig(file_path, dpi=600)

    if show:
        plt.show()
    plt.close()


def plot_rotor_max_speed_mass_torque(folder_path, show=False):
    csv_all_data_file = [folder_path + "Data_PMMA-2mm.csv", folder_path + "Data_PMMA-3mm.csv",
                         folder_path + "Data_PMMA-4mm.csv", folder_path + "Data_PMMA-5mm.csv",
                         folder_path + "Data_PMMA-6mm.csv"]

    average_PMMA_speed_ccw = []
    average_PMMA_speed_cw = []
    std_PMMA_speed_ccw = []
    std_PMMA_speed_cw = []
    average_PMMA_speed = []
    std_PMMA_speed = []
    average_torque_ccw = []
    average_torque_cw = []
    average_torque = []
    std_torque_ccw = []
    std_torque_cw = []
    std_torque = []

    average_power = []
    std_power = []

    PMMA_2mm = 7.51  # in grams
    PMMA_3mm = 10.14
    PMMA_4mm = 12.17
    PMMA_5mm = 17.60
    PMMA_6mm = 19.00
    rotors_mass = [PMMA_2mm, PMMA_3mm, PMMA_4mm, PMMA_5mm, PMMA_6mm]

    cm = 1 / 2.54  # used to convert in centimeters

    for file in csv_all_data_file:
        # Load the data from the "all data csv" file
        df_all_data = pd.read_csv(file, encoding='utf-8')
        PMMA_speed_ccw = df_all_data[(df_all_data["Pmech (mW)"] != 0.0) & (df_all_data["Avg velocity (rad/s)"] > 0.0)] ["Avg velocity (rpm)"]
        PMMA_speed_cw = -1*df_all_data[(df_all_data["Pmech (mW)"] != 0.0) & (df_all_data["Avg velocity (rad/s)"] < 0.0)] ["Avg velocity (rpm)"]

        average_PMMA_speed_ccw.append(np.mean(PMMA_speed_ccw))
        std_PMMA_speed_ccw.append(np.std(PMMA_speed_ccw))
        average_PMMA_speed_cw.append(np.mean(PMMA_speed_cw))
        std_PMMA_speed_cw.append(np.std(PMMA_speed_cw))
        average_PMMA_speed.append(np.mean([*PMMA_speed_ccw, *PMMA_speed_cw]))
        std_PMMA_speed.append(np.std([*PMMA_speed_ccw, *PMMA_speed_cw]))

        PMMA_torque_ccw = df_all_data[(df_all_data["Pmech (mW)"] != 0.0) & (df_all_data["Avg velocity (rad/s)"] > 0.0)] ["Average max torque (mN.m)"]
        PMMA_torque_cw = -1*df_all_data[(df_all_data["Pmech (mW)"] != 0.0) & (df_all_data["Avg velocity (rad/s)"] < 0.0)] ["Average max torque (mN.m)"]
        average_torque_ccw.append(np.mean(PMMA_torque_ccw))
        average_torque_cw.append(np.mean(PMMA_torque_cw))
        std_torque_ccw.append(np.std(PMMA_torque_ccw))
        std_torque_cw.append(np.std(PMMA_torque_cw))
        average_torque.append(np.mean([*PMMA_torque_ccw, *PMMA_torque_cw]))
        std_torque.append(np.std([*PMMA_torque_cw, *PMMA_torque_cw]))

        PMMA_power = df_all_data[df_all_data["Pmech (mW)"] != 0.0]["Pmech (mW)"]
        average_power.append(np.mean(PMMA_power))
        std_power.append(np.std(PMMA_power))

    fig, ax1 = plt.subplots(figsize=(8.8*cm, 6*cm))
    ax1.plot(rotors_mass, average_PMMA_speed, '.-', label='Rotation velocity', color="blue")
    ax1.errorbar(rotors_mass, average_PMMA_speed, yerr=std_PMMA_speed, ls = "None", color='black', capsize=2)

    plt.ylim(-0.1, 6)
    plt.xlim(7.3, 19.5)
    ax1.axvline(x=PMMA_2mm, linestyle="--", color="grey", linewidth=0.5)
    ax1.axvline(x=PMMA_3mm, linestyle="--", color="grey", linewidth=0.5)
    ax1.axvline(x=PMMA_4mm, linestyle="--", color="grey", linewidth=0.5)
    ax1.axvline(x=PMMA_5mm, linestyle="--", color="grey", linewidth=0.5)
    ax1.axvline(x=PMMA_6mm, linestyle="--", color="grey", linewidth=0.5)

    ax1.set_ylabel("Rotation velocity (RPM)")
    ax1.set_xlabel("Rotor mass")
    ax1.set_xticks([PMMA_2mm, PMMA_3mm, PMMA_4mm, PMMA_5mm, PMMA_6mm], ['m1', 'm2', 'm3', 'm4', 'm5'])

    ax3 = ax1.twinx()
    ax3.set_ylabel("Torque (mN.m)")
    ax3.errorbar(rotors_mass, average_torque, yerr=std_torque, ls = "None", color='black', capsize=2)
    ax3.plot(rotors_mass, average_torque, '--.', label='Torque', color='k')
    ax3.set_ylim(0.8, 3.0)

    lines, labels = ax1.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    legend = ax3.legend(lines + lines3, labels + labels3, loc="lower center", borderpad=0.3, borderaxespad=0.3, handlelength=1, labelspacing=0.3)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    plt.tight_layout()

    # Saving filepath
    file_path = folder_path + "Velocity_mass_torque.png"
    plt.savefig(file_path, dpi=600)

    if show:
        plt.show()
    plt.close()


def filter_inst_displacement(y):
    """
    All other calculation will be based on this returned data, so any filter applied here will propagate into
    the rest of the computations
    :param y: The raw instantaneous displacement from the DataFrame series (in mm)
    :return: A filtered array of the displacement (in mm)
    """
    return savgol_filter(y, 15, 3)


def compute_total_displacement(y):
    """
    :param y: The raw instantaneous displacement from the DataFrame series (in mm)
    :return: The cumulative rotor displacement (in mm)
    """
    return y.cumsum()


def compute_velocity(y):
    """
    :param y: The raw instantaneous displacement from the DataFrame series (in mm)
    :return: The rotation speed (in m/s)
    """
    return filter_inst_displacement(y) * camera_frame_rate / 1000  # The speed in m/s


def compute_acceleration(x, y):
    """
    :param x: The dataframe's time series (in s)
    :param y: The raw instantaneous displacement from the DataFrame series (in mm)
    :return: The rotor acceleration in rad/s^-2
    """
    v = compute_velocity(y)
    acc = np.gradient(v, x) / rotor_radius  # The acceleration in rad/s^-2
    return acc


def compute_torque(filename, x, y):
    """
    :param filename: The csv filename
    :param x: The dataframe's time series (in s)
    :param y: The raw instantaneous displacement from the DataFrame series (in mm)
    :return: The torque (in mN.m)
    """
    if filename.find('PMMA-2mm') != -1:
        rotor_mass = PMMA_2mm / 1000  # Make it kg
    elif filename.find('PMMA-3mm') != -1:
        rotor_mass = PMMA_3mm / 1000  # Make it kg
    elif filename.find('PMMA-4mm') != -1:
        rotor_mass = PMMA_4mm / 1000  # Make it kg
    elif filename.find('PMMA-5mm') != -1:
        rotor_mass = PMMA_5mm / 1000  # Make it kg
    elif filename.find('PMMA-6mm') != -1:
        rotor_mass = PMMA_6mm / 1000  # Make it kg
    else:
        logging.warning(f"Rotor type unidentified in {filename}, moment of inertia computation problem.")

    acc = compute_acceleration(x, y)
    I = (1 / 2) * rotor_mass * rotor_radius ** 2  # The moment of inertia in kg.m^2

    return acc * I * 1000 # in mN.m


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global max of dmax-chunks of locals max
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax


def compute_pmech(filename, x, y):
    """
    :param x: The dataframe's time series (in s)
    :param y: The raw instantaneous displacement from the DataFrame series (in mm)
    :param torque: The torque (in mN.m)
    :return: The mechanical power (in mW)
    """
    d = compute_total_displacement(y)
    # Fit lines
    a, b = np.polyfit(x, d, 1)
    torque = compute_torque(filename, x, y)

    return a * torque  # [m/s^-2] * [mN][m]


def compute_torque_max(file_ccw, file_cw):
    print("compute max ccw and cw torques")
    return None, None
