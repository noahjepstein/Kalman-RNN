import numpy as np 
import thLib.imus as IMU
import rawSensorStateProc
import math

SAMPLE_RATE = 10 # hz	

# returns N x 3 ndarray
# N is the number of samples in the trajectory
# X Y and Z position in 3-space at each sample with respect to start position
def getKalmanXYZ(traj): 

	# stub -- doesn't do kalman filtering right now (thus is suceptible to
    # 		  some severe integral drift)

	return rawSensorStateProc.getRawXYZ(traj)


# returns [N, 4] ndarray
# N is the number of samples in the trajectory
# gives a quaternion for each sample that describes the orientation
# of the sensor in 3-space with respect to the starting position
def getKalmanOrientation(traj): 

	# stub -- kalman filtration for orientation finding will be impelemented
	# 		  but for now just uses raw orientation calculations

	quatArr = IMU.kalman_quat(SAMPLE_RATE, traj['accel'], traj['gyro'], traj['mag'])

	return quatArr
