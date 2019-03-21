import librosa
import fnmatch
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pyaudio 
import wave



class Visualizer():
	def __init__(self, sl = 1024):
		self.sampleLength = sl
		'''self.frequencies = {
			16.35 : "C",
			17.32 : "C#",
			18.35 : "D",
			19.45 : "D#",
			20.60 : "E",
			21.83 : "F",
			23.12 : "F#",
			24.50 : "G",
			25.96 : "G#",
			27.50 : "A",
			29.14 : "A#",
			30.87 : "B"
		}'''
		self.frequencies = {
			65.41:'C2', 
			69.30:'C2#',
			73.42:'D2',  
			77.78:'E2b', 
			82.41:'E2',  
			87.31:'F2',  
			92.50:'F2#',
			98.00:'G2', 
			103.80:'G2#',
			110.00:'A2', 
			116.50:'B2b',
			123.50:'B2', 
			130.80:'C3', 
			138.60:'C3#',
			146.80:'D3',  
			155.60:'E3b', 
			164.80:'E3',  
			174.60:'F3',  
			185.00:'F3#',
			196.00:'G3',
			207.70:'G3#',
			220.00:'A3',
			233.10:'B3b',
			246.90:'B3', 
			261.60:'C4', 
			277.20:'C4#',
			293.70:'D4', 
			311.10:'E4b', 
			329.60:'E4', 
			349.20:'F4', 
			370.00:'F4#',
			392.00:'G4',
			415.30:'G4#',
			440.00:'A4',
			466.20:'B4b',
			493.90:'B4', 
			523.30:'C5', 
			554.40:'C5#',
			587.30:'D5', 
			622.30:'E5b', 
			659.30:'E5', 
			698.50:'F5', 
			740.00:'F5#',
			784.00:'G5',
			830.60:'G5#',
			880.00:'A5',
			932.30:'B5b',
			987.80:'B5', 
			1047.00:'C6',
			1109.0:'C6#',
			1175.0:'D6', 
			1245.0:'E6b', 
			1319.0:'E6', 
			1397.0:'F6', 
			1480.0:'F6#',
			1568.0:'G6',
			1661.0:'G6#',
			1760.0:'A6',
			1865.0:'B6b',
			1976.0:'B6', 
			2093.0:'C7'
			} 

	def loadAudio(self, file, sar):
		audio, sr = librosa.load(file, sar, mono = True)
		return audio, sr

	def detectNote(self, ft, duration):
		amp = np.max(ft)
		freq = np.argmax(ft)/duration
		bestDist = freq
		bestFreq = 1661.0
		for k, _ in self.frequencies.items():
			distance = np.abs(freq - k)
			if distance < bestDist:
				bestDist = distance
				bestFreq = k
		return self.frequencies[bestFreq], amp

	def visAudio(self, file, sar = None, dynamicSl = False, time = 1):
		audio, sr = self.loadAudio(file, sar)
		if dynamicSl:
			sl = int(sr*time)
		else:
			sl = self.sampleLength
		

		length = len(audio)
		duration = (float(sl)/float(sr))
		i = 0
		#print(self.sampleLength/sr)
		print("Time duration per sample: %0.4f seconds"%(duration))
		i = 0
		ffts = []
		maxFreqs = []
		while i < length - sl:
			ffts.append(np.log(np.abs(np.fft.rfft(audio[i:i+sl])) + 1e-5))
			maxFreqs.append(np.argmax(ffts[-1])/duration)
			i += int(sl/2)
		plt.imshow(np.array(ffts).transpose(), origin='lower')
		#plt.scatter(range(len(maxFreqs)), maxFreqs, s=0.1, c="red")
		plt.show()
		print(vis.detectNote(ffts[256], duration))

	def testDetector(self, file, sar = None, time=1):
		audio, sr = self.loadAudio(file, sar)
		sl = int(sr*time)
		p = pyaudio.PyAudio()
		w = wave.open(file)
		stream = p.open(format=p.get_format_from_width(w.getsampwidth()),
			channels=w.getnchannels(),
			rate=w.getframerate(),
			output=True
			)

		data = w.readframes(sl)
		i = 0
		prevNote = "C1"
		while data != '':
			# writing to the stream is what *actually* plays the sound.
			stream.write(data)
			data = w.readframes(sl)
			note, amp = self.detectNote(np.log(np.abs(np.fft.rfft(np.blackman(sl)*audio[i:i+sl])) + 1e-5), time)
			if prevNote != note and amp > 1.5:
				print("note: %s, amp %0.4f"%(note, amp))
				prevNote = note
			i += sl

		stream.close()

		p.terminate()		



			


if __name__ == "__main__":
	vis = Visualizer(2**16)
	#vis.visAudio("D:\\MAESTRO\\maestro-v1.0.0\\2017\\MIDI-Unprocessed_046_PIANO046_MID--AUDIO-split_07-06-17_Piano-e_2-02_wav--1.wav", dynamicSl = True, time = 0.02)
	vis.testDetector("D:\\MAESTRO\\maestro-v1.0.0\\2017\\MIDI-Unprocessed_047_PIANO047_MID--AUDIO-split_07-06-17_Piano-e_2-04_wav--4.wav", time = 0.05)

