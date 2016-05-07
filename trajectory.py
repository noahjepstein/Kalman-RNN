import numpy as np 
import math
import os
import pickle
import thLib
import pykalman
	
class Trajectory: 

	def __init__(self, length = 0, forTraining = True 
				 mag = None, accel = None, gyro = None, waypoints = None, 
				 rawPos = None, rawOrient = None, 
				 KalmanPos = None, KalmanOrient = None): 

		self.length = length
		self.forTraining = forTraining
		self.mag = mag
		self.accel = accel
		self.gyro = gyro
		self.waypoints = waypoints
		self.rawPos = rawPos
		self.rawOrient = rawOrient
		self.KalmanPos = KalmanPos
		self.KalmanOrient = KalmanOrient



	