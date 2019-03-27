import librosa
import fnmatch
import threading
import os
import numpy as np
import tensorflow as tf
import options as o
import random



class AudioReader():
	def __init__(self, aDir, sampleRate, receptiveField, coord, sampleSize=None, silenceThreshold=None, queueSize=o.options["batch_size"]):
		self.dir = aDir
		self.coord = coord
		self.sampleRate = sampleRate
		self.receptiveField = receptiveField
		self.sampleSize = sampleSize
		self.silenceThreshold = silenceThreshold
		self.queueSize = queueSize

		self.threads = []
		self.sampleHolder = tf.placeholder(tf.float32, shape=None)
		self.sampleQue = tf.PaddingFIFOQueue(self.queueSize, ['float32'], shapes=[(None, 1)], name="Sample_Queue")
		self.sampleEnque = self.sampleQue.enqueue([self.sampleHolder])

		self.noiseHolder = tf.placeholder(tf.float32, shape=None)
		self.noiseQue = tf.PaddingFIFOQueue(self.queueSize, ['float32'], shapes=[(None, o.options["noise_dimensions"])], name="Noise_Queue")
		self.noiseEnque = self.noiseQue.enqueue([self.noiseHolder])

	def randomFiles(self, files):
		for file in files:
			file_index = random.randint(0, (len(files) - 1))
			yield files[file_index]

	def deque(self, n):
		return self.sampleQue.dequeue_many(n)

	def dequeNoise(self, n):
		return self.noiseQue.dequeue_many(n)

	def findFiles(self, directory, pattern='*.wav'):
		'''Recursively finds all files matching the pattern.'''
		files = []
		for root, dirnames, filenames in os.walk(directory):
			for filename in fnmatch.filter(filenames, pattern):
				files.append(os.path.join(root, filename))
		return files

	def loadAudio(self, dir, sar=None):
		files = self.findFiles(dir)

		print("Files length: {}".format(len(files)))
		randoms = self.randomFiles(files)
		for filename in randoms:
			audio, sr = librosa.load(filename, sr = sar, mono = True)
			audio = audio.reshape(-1,1)	
			yield audio, filename

	def trimSilence(self, audio, frame_length=2048):
		if audio.size < frame_length:
			frame_length = audio.size
		energy = librosa.feature.rmse(audio, frame_length=frame_length)
		frames = np.nonzero(energy > threshold)
		indices = librosa.core.frames_to_samples(frames)[1]
		# Note: indices can be an empty array, if the whole audio was silence.

		return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


	def threadMain(self, sess):
		stop = False
		iters = 0
		while not stop:
			iterator = self.loadAudio(self.dir, sar=self.sampleRate)

			for audio, filename in iterator:
				if self.coord.should_stop():
					stop = True
					break
				if self.silenceThreshold is not None:
					audio = self.trimSilence(audio[:, 0], self.silenceThreshold)
					audio = audio.reshape(-1, 1)
					if audio.size == 0:
						print("Warning: {} was ignored as it contains only "
						"silence. Consider decreasing trim_silence "
						"threshold, or adjust volume of the audio."
						.format(filename))
				audio = np.pad(audio, [[self.receptiveField, 0],[0,0]],'constant')

				if self.sampleSize:
					while len(audio) > self.receptiveField:
						piece = audio[:(self.receptiveField + self.sampleSize), :]
						sess.run(self.sampleQue, feed_dict={self.sampleHolder : piece})
						audio = audio[self.sampleSize:,:]

						noise = np.random.normal(o.options["noise_mean"], o.options["noise_variance"], size=o.options["noisedimensions"]).reshape(-1,1)
						sess.run(self.noiseQue, feed_dict={self.noiseHolder : noise})

	def startThreads(self, sess, nThreads=1):
		for _ in range(nThreads):
			thread = threading.Thread(target=self.threadMain, args=(sess,))
			thread.daemon=True
			thread.start()
			self.threads.append(thread)



if __name__ == "__main__":
	l = AudioReader("maestro-v1.0.0", 160000, 3)
