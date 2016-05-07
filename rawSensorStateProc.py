import numpy as np 
import thLib.imus as IMU
import pykalman
import math
import thLib.rotmat as rotateMatrix

SAMPLE_RATE = 10 # hz

# takes trajectory dictionary object. to be replaced with trajectory 
# class object at a later date. 
# return Nx3 array of raw position estimates w.r.t starting null position
def getRawXYZ(traj): 

	quats, positions = IMU.calc_QPos(traj['initOrient'], traj['gyro'], np.array([0,0,0]), traj['accel'], SAMPLE_RATE)

	return positions



# takes trajectory dictionary object. to be replaced w/ traj class obj
# returns Nx4 array of raw orientation estimates (given as a quaternion w.r.t. origin)
def getRawOrientation(traj): 

	quats, positions = IMU.calc_QPos(traj['initOrient'], traj['gyro'], np.array([0,0,0]), traj['accel'], SAMPLE_RATE)

	return quats

# takes orData, an Nx3 ndarray of orientation values about 
# X, Y, and Z axes in degrees, respective to short dimension of array. 
# converts values to angular velocity in deg/s. 
# outputs an Nx3 ndarray of angular velocity values about 
# X, Y, Z axes respective to the short dimension of the array. 
def orientationToGyro(orData): 

	gyroData = []

	for i, sample in enumerate(orData): 
		if i == 0: 
			gyroData.append([0,0,0])
		else: 
			gyroData.append([(i - j) / float(0.1) for i, j in zip(orData[i], orData[i - 1])])

	return gyroData


# PARAMS: initOrientation -- [1x3] vector that specifies orinetation 
# 		  in degrees in X, Y, and Z directions, respectively. 
# RETURNS: 3x3 matrix that specifies the sensor's initial rotation 
# with respect to the earth's coordinate frame
def calcInitialOrientation(initOrientation): 

	Z = rotateMatrix.R3(initOrientation[2])
	Y = rotateMatrix.R2(initOrientation[1])
	X = rotateMatrix.R1(initOrientation[0])

	return np.dot(np.dot(Z, Y), X)

