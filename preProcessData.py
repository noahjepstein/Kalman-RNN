import numpy as np 
import math
import os
import pickle
import glob
import rawSensorStateProc
# from trajectory import Trajectory

WAYPOINTS_ELEMS_PER_LINE = 6 
UJILocDataFile = os.getcwd() + '/data/lines/c1/l1n_01.txt'

def loadTrajectoryData(inFile = UJILocDataFile):

	with open(UJILocDataFile, 'r') as dataFile: 
		data = dataFile.read()

	# 9-axis IMU data
	# trajectory: dictionary with three elements
	# N is number of samples in the trajectory (data taken at 10Hz)
	# mag: Nx3 numpy array where each line has XYZ mag data
	# gyro: Nx3 numpy array where each line has XYZ gyro vel data
	# accel: Nx3 numpy array where each line has XYZ lin accelerometer data
	segments = data.split("<", 2)
	IMUDataStr = segments[0].split('\n')[:-1]
	magArr = []
	oriArr = []	
	accelArr = []

	for i, lineStr in enumerate(IMUDataStr): 

		lineStr = lineStr.split(' ', 10)[:-1]
		lineStr = [float(x) for x in lineStr]
		magArr.append(lineStr[1:4]) # xyz mag data for sample
		accelArr.append(lineStr[4:7]) # xyz accelerometer data for single samp
		oriArr.append(lineStr[7:10]) # xyz gyro data for sample

	# values initially are given as euler angles which are not good for imu-type calculations. 
	# so we fix em! 	
	gyroArr = rawSensorStateProc.orientationToGyro(oriArr) 
	initOrientationMatrix = rawSensorStateProc.calcInitialOrientation(oriArr[0])

	# IMUData = [{'mag': magArr, 'gyro': gyroArr, 'accel': accelArr}]
	
	# process waypoint data
	# each waypoint consists of a latitude coordinate, longitude coordinate,
	# and index (what IMU dataopoint it represents)
	waypoints = []
	waypointStr = segments[1].split(">", 2)
	numWaypoints = int(waypointStr[0])
	waypointLns = waypointStr[1].lstrip().split('\n')

	for i, lineStr in enumerate(waypointLns): 

		line = lineStr.split(' ', WAYPOINTS_ELEMS_PER_LINE)
		line = [float(x) for x in line]
		
		if i == 0:
			waypoints.append({'lat': line[0], 'long': line[1], 'index': line[4]}) 
		
		waypoints.append({'lat': line[2], 'long': line[3], 'index': line[5]})

		seqLen = line[5]

	
	traj = ({'waypoints': np.array(waypoints), 'mag': np.array(magArr), 'gyro': np.array(gyroArr), 
			 'accel': np.array(accelArr), 'orientSensed': np.array(oriArr), 
			 'initOrient': initOrientationMatrix, 'seqLen': seqLen})

	return traj

# loadTrajectoryData()