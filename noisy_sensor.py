# Kalman Filtering Example
#
# Noisy Sensor
#
# Author: Adam Noel
#

import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

class NoisySensor:
	def __init__(self, sensorValue, noise):
		self.sensorValue = sensorValue
		self.noise = noise
	def getValue(self):
		return (self.sensorValue + np.random.normal(0, 0.5))

class Averager:
	def __init__(self, initialValue):
		self.currentValue = initialValue
		self.numValues = 1
	def updateEstimate(self, newObservation):
		self.numValues = self.numValues + 1
		self.currentValue = (1/float(self.numValues))*newObservation + ((self.numValues - 1)/float(self.numValues))*self.currentValue
	def getValue(self):
		return self.currentValue

class KalmanFilter:
	def __init__(self, shi, Q, H, R, x, P):
		self.shi = shi
		self.Q = Q
		self.H = H
		self.R = R
		self.x = x
		self.P = P
	def updateState(self, newObservation):
		#Now, project to current state
		self.x = self.shi.dot(self.x)
		self.P = self.shi.dot(self.P).dot(np.transpose(self.shi)) + self.Q

		#First, compute the Kalman Gain
		innerKalmanProduct = np.linalg.inv(H.dot(P).dot(np.transpose(H)) + R)
		kalmanGain = self.P.dot(np.transpose(H)).dot(innerKalmanProduct)

		#Next, update the estimate
		self.x = self.x + kalmanGain.dot(np.array(newObservation) - H.dot(self.x))

		#Next, update the error covariance matrix
		sizeP = self.P.shape[0] #Get the shape of the error matrix
		self.P = (np.eye(sizeP) - kalmanGain.dot(H)).dot(self.P)

	def getState(self):
		return np.asarray(self.x)[0][0] ##Convert from matrix to array

numSteps = 100
sensorNoise = 0.5
sensorValue = 3
firstStateEstimate = 2.6
firstErrorEstimate = 0.5

actualValues = []
observedValues = []
averagedValues = [] #Averager to serve as a kalman filter reference
kalmanValues = []


shi = np.matrix([1])
Q = np.matrix([0.00001])
H = np.matrix([1])
R = np.matrix([sensorNoise])
x = np.matrix([firstStateEstimate])
P = np.matrix([firstErrorEstimate])

kalmanFilter = KalmanFilter(shi, Q, H, R, x, P)
noisySensor = NoisySensor(sensorValue, sensorNoise)
averager = Averager(firstStateEstimate)

for i in range(0, numSteps):
	newObservation = noisySensor.getValue()
	actualValues.append(sensorValue)
	observedValues.append(newObservation)
	averager.updateEstimate(newObservation)
	averagedValues.append(averager.getValue())
	kalmanFilter.updateState(newObservation)
	kalmanValues.append(kalmanFilter.getState())

steps = range(0, numSteps)
plt.plot(steps, actualValues, 'r--', label='Actual')
plt.plot(steps, observedValues, 'bs', label='Observed')
plt.plot(steps, averagedValues, 'g^', label='Averaged')
plt.plot(steps, kalmanValues, 'ko', label='Kalman')
plt.legend(loc='lower right')
#plt.plot(steps, actualValues, 'r--', label='Actual', steps, observedValues, 'bs', label='Observed', steps, averagedValues, 'g^', label='Averaged', steps, kalmanValues, 'ko', label='Kalman')
plt.show()

