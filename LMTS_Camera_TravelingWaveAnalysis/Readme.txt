This code is provided jointly with the scientific paper "Permanently magnetized elastomer rotating actuator using traveling waves", by Jean-Baptiste Chossat and Herbert Shea and to be published in the journal Smart Materials and Structures.
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No  101032223.

All data is saved in the Data folder.
The S17_23-03-2023 corresponds to data from sample 17, data having been recorded on the 23rd of March 2023.

All raw data (high speed camera files) can be found at the following repository: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UYUZKV
Each movie files generates 2 csv files. Because there are 900 movies files (see repo), only an excerpt of the 1800 csv files is pushed with the code.
This should allow you to try the code. To get all the csv files, download the video files (HSCamera_movies folder) and run the code.

In this folder:
-each analyzed video files data is transcribed into two CSV files following this format:
    1) raw movie file name + "csv" for the soft elastomer position through time, based on a single marker position ;
    2) raw movie file name + "_rot.csv" for the position of the PMMA through time, based on PMMA makers position.

-each PMMA plate thickness averaged data is saved following this format: "Data_PMMA-Xmm.csb" where X is the plate
thickness. The files provided in the git repository are already populated with all the data from the analysis of all the
videos.
-the main graphs are also saved there


Other sub-folder are used as follows:

-Torque contains the graphs used to check the torque envelope and torque average value measured
-RotorSpeed contains the graphs or the rotor (PMMA plate) instantaneous speed thorough the experiment
-RotorAcc contains the graphs of the rotor (PMMA plate) instantaneous acceleration thorough the experiment
- Review_movies contains movies generated from the raw files during marker tracking. These files are useful for checking
that the algorithm actually recognizes and tracks the markers well.
-Marker trajectory contains graphs and gifs pertaining to tracking the soft elastomer marker in XY axes and through time.
1) The GIF subfolder contains gifs of the marker through time
2) The Partial_trajectories contains graphs of the position of the marker, color coded to show the marker trajectory over a single time period.
- HSCamera_movies contains the raw video files from the high speed camera. The name format is as follows:
Sample name ; Thickness of plate used ; Experiment total time ; Actuation frequency ; Actuation standing wave phase
difference ; Wave generator amplitude ; phase tracking number (not relevant) ; phase experiment number.
Example:
s17_PMMA-3mm_1.2sec_70Hz_90deg_2.5V_10_5.avi
Corresponds to sample 17, equipped with the 3mm PMMA plate as load, the experiment lasted 1.2 seconds, 70Hz was used as
excitation frequency, the two standing waves are phased out by 90 degrees and have a 2.5V amplitude (total voltage
amplitude used at the coils is 5V because my current amplifiers have a gain of 2), and this is the 5th experiment with
these particular conditions.