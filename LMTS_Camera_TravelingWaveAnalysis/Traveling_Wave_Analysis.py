from Movie_Analysis import Analyze_movie
import logging
import pandas as pd
from os.path import exists
from tqdm import tqdm
import numpy as np
import math
import scipy
from scipy.signal import savgol_filter
import threading
import glob
import imageio.v2 as imageio
from natsort import natsorted

csv_filenames = []
gif_images = []

def traveling_wave_video_analysis(movie_filenames, folder_path, all_data_mode = False, save_data = True):

    for movie_filename in tqdm(movie_filenames, desc="Traveling wave analysis, analyzing movies"):
        csv_filenames.append(movie_filename.replace('.avi', '.csv'))
        # Generates the csv data if the file does not already exist
        # if not exists(csv_filenames[-1]):
        marker_issues, too_many_rect, too_few_rect, frame_numb = Analyze_movie(movie_filename, folder_path,
                                                                               all_data_mode, save_data, False)
        if (marker_issues or too_many_rect or too_few_rect) > 1:
            logging.warning(f"Found {marker_issues} ({100*marker_issues/frame_numb}%) frames w. missing marker.\n "
                            f"Found {too_many_rect} ({100*too_many_rect/frame_numb}%)frames w. more than 1 rectangle.\n"
                            f"Found {too_few_rect} ({100*too_few_rect/frame_numb}%)frames w. no rectangles.")


def traveling_wave_data_analysis(folder_path, csv_rotor_file_list, exp_param, id_string):
    csv_save_path = folder_path + "Data_" + id_string + ".csv"
    max_sequence_number = 0

    df = pd.DataFrame(columns=["Filename", "Sequence #","Rotor", "Actuation duration (s)", "Voltage (V)",
                               "Phase diff (deg)", "Freq (Hz)", "Avg velocity (mm/s)", "Avg velocity (rpm)",
                               "Avg velocity (rad/s)", "Average max torque (mN.m)", "Pmech (mW)"])

    for file in tqdm(csv_rotor_file_list, desc="Traveling wave analysis, analyzing data"):
        # Load this particular file's data
        df_rotor = pd.read_csv(file, encoding='utf-8')
        x = df_rotor['Time (s)']
        y = df_rotor['Delta_x (mm)']
        d = y.cumsum()

        # Fit lines, velocity computation
        a, b = np.polyfit(x, d, 1)  # a is in mm/s now
        velocity_rpm = float((a * 60) / (1000 * 2 * math.pi * exp_param[7]))
        velocity_rad = float(a / (1000 * exp_param[7]))

        filename_parts = file.split('_')
        if filename_parts[9]:
            sequence_number = int(filename_parts[9])
            if max_sequence_number < sequence_number :
                max_sequence_number = sequence_number
        else:
            sequence_number = 0
            max_sequence_number = sequence_number


        df.loc[csv_rotor_file_list.index(file)] = [file,
                                                   sequence_number,
                                                   filename_parts[3],
                                                   filename_parts[4],
                                                   float(filename_parts[7].replace('V', '')),
                                                   int(filename_parts[6].replace('deg', '')),
                                                   filename_parts[5].replace('Hz', ''),
                                                   a,
                                                   velocity_rpm,
                                                   velocity_rad,
                                                   0.0,
                                                   0]


    for sequence in range(1, max_sequence_number+1):
        # Find at which two phase diff the rotor has the highest rotational velocity
        ccw_index = df[df['Sequence #'] == sequence]['Avg velocity (rad/s)'].idxmin()
        cw_index = df[df['Sequence #'] == sequence]['Avg velocity (rad/s)'].idxmax()
        file_torque_ccw = csv_rotor_file_list[ccw_index]
        file_torque_cw = csv_rotor_file_list[cw_index]

        for file in csv_rotor_file_list:
            # Load this particular file's data
            df_rotor = pd.read_csv(file, encoding='utf-8')
            x = df_rotor['Time (s)']
            y = df_rotor['Delta_x (mm)']

            if file == file_torque_ccw:
                # Torque computation
                torque = compute_torque(file, x, y, exp_param)
                torque_max_avg_ccw = compute_torque_max(torque, df_rotor, exp_param, rotation_direction='ccw')
                df.iloc[csv_rotor_file_list.index(file), df.columns.get_loc("Average max torque (mN.m)")] = torque_max_avg_ccw
                df.iat[csv_rotor_file_list.index(file), df.columns.get_loc("Pmech (mW)")] = \
                    torque_max_avg_ccw * df['Avg velocity (rad/s)'].iloc[csv_rotor_file_list.index(file)]

            elif file == file_torque_cw:
                torque = compute_torque(file, x, y, exp_param)
                torque_max_avg_cw = compute_torque_max(torque, df_rotor, exp_param, rotation_direction='cw')
                df.iloc[csv_rotor_file_list.index(file), df.columns.get_loc("Average max torque (mN.m)")] = torque_max_avg_cw
                df.iat[csv_rotor_file_list.index(file), df.columns.get_loc("Pmech (mW)")] = \
                    torque_max_avg_cw * df['Avg velocity (rad/s)'].iloc[csv_rotor_file_list.index(file)]

    df.to_csv(csv_save_path, index=False, sep=',')


def compute_torque_max(torque, df, exp_param, rotation_direction):
    torque_max_avg = 0

    # Shorten the data time series to only actuation time
    new_series_length = len(df[df['Time (s)'] < exp_param[1]]['Time (s)'])

    # Shorten the data torque series to only actuation time
    torque_tmp = torque[:new_series_length]
    # Create the signal's envelope
    envelope_min, envelope_max = hl_envelopes_idx(torque_tmp, dmin=25, dmax=25)

    if rotation_direction == 'cw':
        # Find the peaks here *deprecated*
        # torque_max_index = scipy.signal.find_peaks(torque_tmp, distance= 0.75 * exp_param[0] / (exp_param[8]))

        # Find the average of the torque envelope
        torque_max_avg = np.mean(torque_tmp[envelope_max])

    elif rotation_direction == 'ccw':
        # Find the peaks here *deprecated*
        # torque_max_index = scipy.signal.find_peaks(-torque_tmp, distance= 0.75 * exp_param[0] / (exp_param[8]))

        # Find the average of the torque envelope
        torque_max_avg = np.mean(torque_tmp[envelope_min])

    return torque_max_avg

def compute_torque(filename, x, y, exp_param):
    """
    :param filename: The csv filename
    :param x: The dataframe's time series (in s)
    :param y: The raw instantaneous displacement from the DataFrame series (in mm)
    :return: The torque (in mN.m)
    """
    if filename.find('PMMA-2mm') != -1:
        rotor_mass = exp_param[2] / 1000  # Make it kg
        logging.info("Moment of inertia: PMMA-2mm")
    elif filename.find('PMMA-3mm') != -1:
        rotor_mass = exp_param[3] / 1000  # Make it kg
        logging.info("Moment of inertia: PMMA-3mm")
    elif filename.find('PMMA-4mm') != -1:
        rotor_mass = exp_param[4] / 1000  # Make it kg
        logging.info("Moment of inertia: PMMA-4mm")
    elif filename.find('PMMA-5mm') != -1:
        rotor_mass = exp_param[5] / 1000  # Make it kg
        logging.info("Moment of inertia: PMMA-5mm")
    elif filename.find('PMMA-6mm') != -1:
        rotor_mass = exp_param[6] / 1000  # Make it kg
        logging.info("Moment of inertia: PMMA-6mm")
    else:
        logging.warning(f"Rotor type unidentified in {filename}, moment of inertia computation problem.")

    acc = compute_acceleration(x, y, exp_param)  # The acceleration in rad/s^-2
    I = (1 / 2) * rotor_mass * exp_param[7] ** 2  # The moment of inertia in kg.m^2

    return acc * I * 1000 # in mN.m


def compute_acceleration(x, y, exp_param):
    """
    :param x: The dataframe's time series (in s)
    :param y: The raw instantaneous displacement from the DataFrame series (in mm)
    :return: The rotor acceleration in rad/s^-2
    """
    v = compute_velocity(y, exp_param)
    acc = np.gradient(v, x) / exp_param[7]  # The acceleration in rad/s^-2
    return acc


def compute_velocity(y, exp_param):
    """
    :param y: The raw instantaneous displacement from the DataFrame series (in mm)
    :return: The rotation speed (in m/s)
    """
    v = filter_inst_displacement(y) * exp_param[0] / 1000  # The speed in m/s
    return v


def filter_inst_displacement(y):
    """
    All other calculation will be based on this returned data, so any filter applied here will propagate into
    the rest of the computations
    :param y: The raw instantaneous displacement from the DataFrame series (in mm)
    :return: A filtered array of the displacement (in mm)
    """
    return savgol_filter(y, 15, 3)


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    # Code reused from the following stack overflow post:
    # https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal
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
