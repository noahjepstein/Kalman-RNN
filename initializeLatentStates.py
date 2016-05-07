# initializeLatentStates.py
# Author: Noah J Epstein
# noahjepstein@gmail.com
# Modified: 5/5/2016
# 
# > Processes 9-axis IMU trajectory data 
#   to give inital estimates and kalman filtered estimates for 
#   walking trajectories through hallways in the UJILoc-Mag dataset.
# 
# > Adds new duplicate trajectories that are perturbed with noise
#   so that the dataset size is expanded and more robust due to 
#   additional noise inclusion. 
# 
# > Saves data in pickled format so that it can easily be used 
#   in batch format for training a recurrent neural network. 
#   (See readme or kalmaRNN.py for details)
# 

import numpy as np 
import cPickle
import os	
import glob
from pykalman import KalmanFilter
from preProcessData import loadTrajectoryData
import rawSensorStateProc
import KalmanSensorStateProc

TRAINING_DATAFILE = 'trainingBin.pickle'
TESTING_DATAFILE = 'testingBin.pickle'
IMU_NUM_AXES = 9

trajFileDirTrain = os.getcwd() + '/data/*/*/*.txt'
trajFileDirTest = os.getcwd() + '/data/*/*.txt'
trajectoryFileList = np.array(glob.glob(trajFileDirTrain))
trajectoryFileList = np.append(trajectoryFileList, glob.glob(trajFileDirTest))

trajectoryData = [] # list of trajectories
trajectoryDataTest = [] # just the ones for testing

minSeqLen = 10000000


# arguments: trajectory object with imu, position, orientation, metadata
# RETURNS: same trajectory with the imu data perturbed by gaussian noise
def addNoise(traj, noiseType): 

	nMult = np.random.uniform(low = 0.0, high = 0.1)

	# for each measurement axis sequence, we perturb the values 
	# based on the standard deviation of each measurement in the
	# range of samples

	for i in range(IMU_NUM_AXES): 
	

		# add zero mean gaussian noise in proportion with the standard deviation 
		# on that measurement axis

		if noiseType == "gaussian": 
			noise = np.random.normal(loc = 0.0, 
									 scale = nMult * np.std(traj['nnInput'][:, i]), 
									 size = traj['nnInput'][:, i].shape )
		elif noiseType == 'laplace': 

			noise = np.random.laplace(loc = 0.0, 
									  scale = nMult * np.std(traj['nnInput'][:, i]), 
									  size = traj['nnInput'][:, i].shape )

		elif noiseType == 'multi': 
			
			noise = np.random.normal(loc = 0.0, 
						 			 scale = nMult * np.std(traj['nnInput'][:, i]), 
						 			 size = traj['nnInput'][:, i].shape )
			
			noise = np.add(
						np.random.laplace(loc = 0.0, 
									 scale = nMult * np.std(traj['nnInput'][:, i]), 
									 size = traj['nnInput'][:, i].shape ), 
						noise)
		else: 
			noise = np.zeros_like(traj['nnInput'][:, i])


		traj['nnInput'][:,i] = np.add(traj['nnInput'][:,i], noise)

	return traj


##################################################################################
############################## estimation stuff starts here ###################### 
################################################################################## 

nTraj = 0

for trajFilename in trajectoryFileList:
	
	traj = loadTrajectoryData(trajFilename)
	print trajFilename
	
	# For the LORD will pass through to smite the Egyptians; and when He sees 
	# the blood on the lintel and on the two doorposts, the LORD will pass over 
	# the door and will not allow the destroyer to come in to your houses to smite you.
	if '/tests/' in trajFilename:
		traj['testFile'] = True
	else:
		traj['testFile'] = False

	traj['rawXYZ'] = rawSensorStateProc.getRawXYZ(traj)
	traj['rawOrientEst'] = rawSensorStateProc.getRawOrientation(traj)
	traj['KalmanXYZ'] = KalmanSensorStateProc.getKalmanXYZ(traj)
	traj['KalmanOrientEst'] = KalmanSensorStateProc.getKalmanOrientation(traj)



	imuRaw = []

	# setup plain NN input vectors for RNN training
	for i, magSamp in enumerate(traj['mag']): 

		sample = np.concatenate((magSamp, traj['gyro'][i], traj['accel'][i]))
		imuRaw.append(sample)

	imuRaw = np.array(imuRaw)
	traj['nnInput'] = imuRaw


	if traj['testFile']: 
		trajectoryDataTest.append(traj)
	else: 
		trajectoryData.append(traj)
	
	nTraj += 1

	# add 20 noisy input sequences to increase the size of the training set 
	# and add variability to the training set ot make it more robust
	# just gaussian noise for now, but could add numerous distributions
	# to create additional training data
	for i in range(60):
		if traj['testFile']: 
			trajectoryDataTest.append(addNoise(traj, 'gaussian'))
		else: 
			trajectoryData.append(addNoise(traj, 'gaussian'))
		nTraj += 1

	print nTraj

# save to avoid doing processing again
cPickle.dump(trajectoryData, open(TRAINING_DATAFILE, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(trajectoryDataTest, open(TESTING_DATAFILE, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)