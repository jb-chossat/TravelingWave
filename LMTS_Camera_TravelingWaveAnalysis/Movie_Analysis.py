import pandas as pd
# import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from tqdm import tqdm
import os
import logging
logging.basicConfig(level=logging.INFO)

# Blob detection specs
########################################################################################################################
# Setup SimpleBlobDetector parameters.
params = cv.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200
# Filter by Area.
params.filterByArea = True
params.minArea = 650
params.maxArea = 1000
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.65
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
params.filterByColor = False
ver = (cv.__version__).split('.')  # Check version or your code will segfault...

# Parameters for the circle detection
if int(ver[0]) < 3:
    detector = cv.SimpleBlobDetector(params)
else:
    detector = cv.SimpleBlobDetector_create(params)

params.minArea = 180
params.maxArea = 400
if int(ver[0]) < 3:
    detector_rotor = cv.SimpleBlobDetector(params)
else:
    detector_rotor = cv.SimpleBlobDetector_create(params)
########################################################################################################################
crop_x = 64  # cannot be more than 127, more means thinner image
crop_y = 254   # cannot be more than 254, more means taller image
crop_rotor_x = 8
crop_rotor_y = 190
rotor_image_threshold = 130
marker_image_threshold = 110

def Analyze_movie(filename, folder_path, all_data_mode=False, save_data=False, save_graph=False):
    """
    Analyzes the movie.
    :param filename: The path to the movie.
    :param all_data_mode: False for testing the algorithm (100 times smaller data)
    :param save_data: True to save the dataframe of the raw displacement data
    :param save_graph: True to graph / save the graphs of the raw displacement data
    :return:
    """

    # High speed camera specs & variables
    ####################################################################################################################
    # control_movie_name = filename.replace(".avi", "/ReviewMovie/") + "_control.avi"
    # control_rotor_movie_name = filename.replace(".avi", "/ReviewMovie/") + "_r_control.avi"

    if not os.path.exists(folder_path + "Review_movies/"):
        os.makedirs(folder_path + "Review_movies/")
    control_movie_name = (folder_path + "Review_movies/" + filename.split("/")[-1]).replace(".avi", "_control.avi")
    control_rotor_movie_name =  (folder_path + "Review_movies/" + filename.split("/")[-1]).replace("avi", "_r_control.avi")
    csv_filename = folder_path + filename.split("/")[-1].replace(".avi", ".csv")
    csv_filename2 = folder_path + filename.split("/")[-1].replace(".avi", "_rot.csv")

    framerate = 4000  # The number of frames per sec
    marker_size = 1.8  # The marker size in mm
    marker_x = 0.0  # Relative position of the marker in x, at t=0
    marker_y = 0.0  # Relative position of the marker in y, at t=0
    calibration = 0.0  # Done using the average marker size in px and the known marker size in mm
    movie_time = 0.0  # Used to iterate the time series for each new frame

    no_rectangle_issue = 0
    too_many_rectangles_issue = 0
    marker_frame_issues = 0

    movie = cv.VideoCapture(filename)
    _, frame = movie.read()
    frame_h, frame_w, frame_c = frame.shape[:3]
    frame_number = int(movie.get(cv.CAP_PROP_FRAME_COUNT))
    logging.info(f"Frame specs: h={frame_h}, w={frame_w}, and {frame_c} channel(s).")
    logging.info(f"Movie's total frame number = {frame_number}")
    hsc_data = pd.DataFrame(columns=['Time (s)', 'blob_x (px)', 'blob_y (px)', 'blob_size (px)'])
    hsc_data2 = pd.DataFrame(columns=['Time (s)', 'R_coordinates (px)', 'Delta_x (px)', 'Delta_y (px)',
                                      'Delta_x (mm)', 'Delta_y (mm)'])

    video_writer = cv.VideoWriter(control_movie_name, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5,
                                  (128, crop_y + 1))

    # Short run if only testing
    if all_data_mode:
        max_frame_number = frame_number
    else:
        max_frame_number = int(frame_number / 10)

# Main marker analysis loop
########################################################################################################################
    logging.info(f"File: {filename}")
    for frame in tqdm(range(max_frame_number), desc="Movie analysis: marker tracking", mininterval=5):
        movie.set(cv.CAP_PROP_POS_FRAMES, frame)
        success, image = movie.read()

        # 1) Make the image grayscale (and uint8 array)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 2) Crop the image, [height, width]
        # First image is for marker tracking, second image is for rotor tracking (assumption: rotor always above marker)
        cropped_image = gray_image[0:crop_y, crop_x:frame_w - crop_x]
        # 3) Threshold the 2 images
        th_value, th_image = cv.threshold(cropped_image, marker_image_threshold, 255, cv.THRESH_TOZERO)
        # 4) Find the circle in the image
        keypoint = detector.detect(th_image)
        im_with_keypoints = cv.drawKeypoints(th_image, keypoint, np.array([]), (0, 0, 255),
                                             cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # 6) Check that there is a single blob per image
        if len(keypoint) > 1 :
            logging.warning(f"More than one marker found in frame sequence at t = {movie_time}.")
            marker_frame_issues += 1

        elif len(keypoint) < 1 :
            logging.warning(f"No marker found in frame sequence at t = {movie_time}.")
            marker_frame_issues += 1

        else:
            # 7) Populate marker origins
            if marker_x == 0.0 and marker_y == 0.0:
                marker_x = cv.KeyPoint_convert(keypoint)[0][0]  # Marker original position in x, in pixels
                marker_y = cv.KeyPoint_convert(keypoint)[0][1]  # Marker original position in y, in pixels
            # 8) Add the data to the dataframe
            hsc_data.loc[len(hsc_data)] = \
                ([movie_time, cv.KeyPoint_convert(keypoint)[0][0],
                  cv.KeyPoint_convert(keypoint)[0][1], keypoint[0].size])
            # Generate bogus data
            # hsc_data.loc[len(hsc_data)] = ([movie_time, 0, 0, 0])

        # 9) Assemble the final images in a new movie for checking purposes
        if save_data:
            video_writer.write(im_with_keypoints)
        # 10) Update the time series
        movie_time += 1 / framerate
    video_writer.release()
########################################################################################################################
    logging.info(f"Mean blob size (px): {hsc_data['blob_size (px)'].mean()}")
    calibration = marker_size / hsc_data['blob_size (px)'].mean()
    logging.info(f"Blob size variance (μm): {hsc_data['blob_size (px)'].var() * calibration * 1000}")
    logging.info(f"Calibration: {calibration} mm per px")
    # Moving up is an increase in Y, while moving right is an increase in X
    hsc_data['blob_x (mm)'] = (hsc_data['blob_x (px)'] - marker_x) * calibration
    hsc_data['blob_y (mm)'] = (marker_y - hsc_data['blob_y (px)']) * calibration

    # Reset video writer variables
    video_writer_rotor = cv.VideoWriter(control_rotor_movie_name, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5,
                                  (frame_w - 2 * crop_rotor_x, crop_rotor_y + 1))
    movie_time = 0.0  # Used to iterate the time series for each new frame
########################################################################################################################
    for frame in tqdm(range(max_frame_number), desc="Movie analysis, rotor tracking", mininterval=5):

        movie.set(cv.CAP_PROP_POS_FRAMES, frame)
        success, image2 = movie.read()

        # 1) Make the image grayscale (and uint8 array)
        gray_image = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        # 2) Crop the image, [height, width]
        # First image is for marker tracking, second image is for rotor tracking (assumption: rotor always above marker)
        cropped_image_rotor = gray_image[0:crop_rotor_y, crop_rotor_x:frame_w - crop_rotor_x]

        # 3) Sharpen the image contours for better detection
        image_sharp = unsharp_mask(cropped_image_rotor, sigma=1.0, amount=1.0, threshold=100)

        # 4) Threshold the image
        th_value_rotor, th_image_rotor = cv.threshold(image_sharp, rotor_image_threshold, 255, cv.THRESH_TOZERO)

        # 5) Find all the blobs, draw their contours
        keypoints_rotor = detector_rotor.detect(th_image_rotor)
        im_with_keypoints = cv.drawKeypoints(th_image_rotor, keypoints_rotor, np.array([]), (0, 255, 0),
                                             cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # 6) Make a list of the blobs coordinates
        disk_list = []
        for disk in range(0, len(keypoints_rotor)):
            x = cv.KeyPoint_convert(keypoints_rotor)[disk][0]
            y = cv.KeyPoint_convert(keypoints_rotor)[disk][1]
            disk_list.append((x,y))
        sorted_disks = sorted(disk_list)

        delta = 5  # The allowed pixel difference between one rectangle and the same rectangle on next frame
        blob_motion = []
        blob_avg_motion = (0.0, 0.0)
        blob_found = 0

        # 7) Now track the 6 innermost blobs' positions and draw their centers
        if len(hsc_data2) > 0:
            for blob_index in range(6):
                previous_blob_x = hsc_data2.iloc[-1]['R_coordinates (px)'][blob_index][0]
                previous_blob_y = hsc_data2.iloc[-1]['R_coordinates (px)'][blob_index][1]
                # If the new blob position closely matches the old one,
                # then we have no problem following the blob motion
                if (previous_blob_x + delta > sorted_disks[blob_index + 1][0] > previous_blob_x - delta) and \
                                    (previous_blob_y + delta > sorted_disks[blob_index+1][1] > previous_blob_y - delta):
                    blob_found += 1
                    # Moving right should be seen as an increase in X position
                    blob_motion.append((sorted_disks[blob_index + 1][0] - previous_blob_x,
                                                                   previous_blob_y - sorted_disks[blob_index+1][1]))

                    blob_avg_motion = (blob_avg_motion[0] + blob_motion[-1][0], blob_avg_motion[1] + blob_motion[-1][1])

                    # Draw the selected disks centers, this is an approximation since the real center is not an integer!
                    cv.circle(im_with_keypoints,
                              (int(sorted_disks[blob_index+1][0]), int(sorted_disks[blob_index+1][1])), 1,
                              (255, 255, 255), 1)

                # In that case, the new blob position and the previous one do not correspond.
                # This is likely because a new blob appeared / and old one got out of frame.
                # As a consequence, the order the blob coordinates take in the list is shifted.
                # I need to match 5 of the rectangles with their respective coordinates in the previous frame to still
                # get an approximate value of displacement.
                else:
                    if blob_index < 5:
                        if(previous_blob_x + delta > sorted_disks[blob_index + 2][0] > previous_blob_x - delta) \
                        and (previous_blob_y + delta > sorted_disks[blob_index + 2][1] > previous_blob_y - delta):

                            blob_found += 1
                            blob_motion.append((sorted_disks[blob_index + 2][0] - previous_blob_x,
                                                previous_blob_y - sorted_disks[blob_index + 2][1]))
                            blob_avg_motion = (blob_avg_motion[0] + blob_motion[-1][0],
                                               blob_avg_motion[1] + blob_motion[-1][1])

                            # Draw the selected disks centers, this is an approximation since the real center is not an integer!
                            cv.circle(im_with_keypoints,
                                      (int(sorted_disks[blob_index + 2][0]), int(sorted_disks[blob_index + 2][1])),
                                      1, (255, 255, 255), 1)

                    elif blob_index > 0:
                        if (previous_blob_x + delta > sorted_disks[blob_index][0] > previous_blob_x - delta) and \
                            (previous_blob_y + delta > sorted_disks[blob_index][1] > previous_blob_y - delta):

                            blob_found += 1
                            blob_motion.append((sorted_disks[blob_index][0] - previous_blob_x,
                                                previous_blob_y - sorted_disks[blob_index][1]))
                            blob_avg_motion = (blob_avg_motion[0] + blob_motion[-1][0],
                                               blob_avg_motion[1] + blob_motion[-1][1])

                            # Draw the selected disks centers
                            cv.circle(im_with_keypoints,
                                      (int(sorted_disks[blob_index][0]), int(sorted_disks[blob_index][1])), 1,
                                      (255, 255, 255), 1)
                    else:
                        logging.warning(f"No blob rotor found (n={blob_found} at t = {movie_time}), filename: {filename}")

            blob_avg_motion = (blob_avg_motion[0] / blob_found, blob_avg_motion[1] / blob_found)

        blob_avg_motion_mm = (blob_avg_motion[0] * calibration, blob_avg_motion[1] * calibration)

        # 8) Store the coordinates and the average instantaneous motion of the 6 blobs in pixels and mm
        hsc_data2.loc[len(hsc_data2)] = (movie_time, sorted_disks[1:7], blob_avg_motion[0], blob_avg_motion[1],
                                                 blob_avg_motion_mm[0], blob_avg_motion_mm[1])

        # 9) Assemble the final images in a new movie for checking purposes
        if save_data:
            video_writer_rotor.write(im_with_keypoints)
        # 10) Update the time series
        movie_time += 1 / framerate
    video_writer_rotor.release()
########################################################################################################################
    # Save all the data in a csv file
    if save_data:
        hsc_data.to_csv(csv_filename, index=False, encoding='utf-8')
        hsc_data2.to_csv(csv_filename2, index=False, encoding='utf-8')
########################################################################################################################

    return marker_frame_issues, too_many_rectangles_issue, no_rectangle_issue, frame_number

# https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
