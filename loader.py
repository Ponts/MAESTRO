import librosa
import fnmatch
import threading
import os
import numpy as np
import tensorflow as tf
import options as o
import random



class AudioReader():
	def __init__(self, aDir, sampleRate, receptiveField, sampleSize=None, silenceThreshold=None, queueSize=o.options["batch_size"]):
		self.dir = aDir
		self.sampleRate = sampleRate
		self.receptiveField = receptiveField
		self.sampleSize = sampleSize
		self.silenceThreshold = silenceThreshold
		self.queueSize = queueSize

		self.threads = []
		self.sampleHolder = tf.placeholder(tf.float32, shape=None)
		self.sampleQue = tf.PaddingFIFOQueue(self.queueSize, ['float32'], shapes=[(None, 1)], name="Sample_Queue")
		self.sampleEnque = self.sampleQue.enqueue([self.sampleHolder])

		self.noiseHolder = tf.placeholder(tf.float32, shape=())
		self.noiseQue = tf.PaddingFIFOQueue(self.queueSize, ['float32'], shapes=[()], name="Noise_Queue")
		self.noiseEnque = self.noiseQue.enqueue([self.noiseHolder])

	def randomFiles(self, files):
		for file in files:
			file_index = random.randint(0, (len(files) - 1))
			yield files[file_index]

	def findFiles(self, directory, pattern='*.wav'):
		'''Recursively finds all files matching the pattern.'''
		files = []
		for root, dirnames, filenames in os.walk(directory):
			for filename in fnmatch.filter(filenames, pattern):
				files.append(os.path.join(root, filename))
		return files

	def loadAudio(self, dir):
		files = self.findFiles(dir)

		print("Files length: {}".format(len(files)))
		randoms = self.randomFiles(files)
		for filename in randoms:
			audio, sr = librosa.load(filename, sr = None, mono = True)
			audio = audio.reshape(-1,1)	
			yield audio, filename

	def threadMain(self, sess):
		stop = False
		iters = 0
		while not stop:
			iterator = self.loadAudio(self.dir)

			iters += 1
			if iters > 10:
				stop = True


if __name__ == "__main__":
	l = AudioReader("maestro-v1.0.0", 160000, 3)
