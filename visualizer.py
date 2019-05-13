import librosa, librosa.display
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
			27.5000:'A0',
			29.1352:'B0b',
			30.8677:'B0', 
			32.7032:'C1',
			34.6478:'C1#',
			36.7081:'D1',
			38.8909:'E1b',
			41.2034:'E1', 
			43.6535:'F1',
			46.2493:'F1#',
			48.9994:'G1',
			51.9131:'G1#',
			55.00:'A1',
			58.2705:'B1b',
			61.7354:'B1',
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
			2093.0:'C7',
			2217.46 : 'C7#',
			2349.32 : 'D7',
			2489.02 : 'E7b',
			2637.02 : 'E7',
			2793.83 : 'F7',
			2959.96 : 'F7#',
			3135.96 : 'G7',
			3322.44 : 'G7#',
			3520.00 : 'A7',
			3729.31 : 'B7b',
			3951.07 : 'B7',
			4186.01 : 'C8',
			} 

	def loadAudio(self, file, sar):
		audio, sr = librosa.load(file, sar, mono = True)
		return audio, sr

	def detectNote(self, audio, duration):
		ft = np.log(np.abs(np.fft.rfft(np.blackman(len(audio))*audio)) + 1e-5)
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
		print(len(audio))
		#print(audio[int(len(audio)/2):int(len(audio)/2)+300])
		plt.plot(audio)#[int(len(audio)/2):int(len(audio)/2)+5147])
		plt.plot(audio[:5117])
		plt.show()
		return

	def testDetector(self, file, sar = None, time=1):
		audio, sr = self.loadAudio(file, sar)
		#librosa.output.write_wav("Generated/gangenwave.wav", audio, sr, norm=True, dtype=np.int16)
		sl = int(sr*time)
		p = pyaudio.PyAudio()
		w = wave.open(file, 'rb')
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
			le = len(audio[i:i+sl])
			if le < 1:
				break
			note, amp = self.detectNote(audio[i:i+sl], time) # Move this to the detect note function
			if prevNote != note and amp > 1.:
				print("note: %s, amp %0.4f"%(note, amp))
				prevNote = note
			i += sl

		stream.close()

		p.terminate()		


	def compare(self, song1, song2, sar1=None, sar2=None):
		audio1, sr1 = self.loadAudio(song1, sar1)
		audio2, sr2 = self.loadAudio(song2, sar2)

		plt.plot(audio1[5117:-1])#[int(len(audio)/2):int(len(audio)/2)+5147])
		plt.plot(audio2[5117:-1])#[int(len(audio)/2):int(len(audio)/2)+5147])

		diff = np.average(audio1[5117:-1] - audio2[5117:-1])
		print(diff)
		plt.show()

	def mel_spectogram(self, file, sar, title="Generated"):
		audio, sr = librosa.load(file, sar)
		if (len(audio)) > sar:
			ra = np.random.randint(0,len(audio)-sr)
			audio = audio[ra:ra+sr]
		print(len(audio))
		print(sr)
		S = librosa.feature.melspectrogram(y=audio, sr=sr)
		plt.figure(figsize=(10, 4))
		librosa.display.specshow(librosa.power_to_db(S,
													ref=np.max),
													y_axis='mel', fmax=sr/2,
													x_axis='time')
		plt.colorbar(format='%+2.0f dB')
		plt.title(title)
		plt.tight_layout()
		
			


if __name__ == "__main__":
	vis = Visualizer(2**16)
	#vis.visAudio("D:\\MAESTRO\\maestro-v1.0.0\\2017\\MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--2.wav", dynamicSl = True, time = 0.02)
	#vis.visAudio("D:\\MAESTRO\\Generated\\gangen.wav", dynamicSl = True, time=0.02)
	#vis.visAudio("D:\\MAESTRO\\Generated\\visualizedilated_stack5.wav", dynamicSl = True, time=0.02)
	#vis.visAudio("D:\\normal_wavenet\\generate.wav", dynamicSl = True, time=0.02)
	#vis.testDetector("D:\\MAESTRO\\maestro-v1.0.0\\2017\\MIDI-Unprocessed_047_PIANO047_MID--AUDIO-split_07-06-17_Piano-e_2-04_wav--4.wav", time = 0.05)
	#vis.compare("D:\\MAESTRO\\Generated\\gangen.wav", "D:\\MAESTRO\\Generated\\bla.wav")
	vis.testDetector("D:\\MAESTRO\\Generated\\gangen2.wav", time = 0.05)
	#vis.mel_spectogram("D:\\MAESTRO\\Generated\\firstModel\\1secGAN998849.wav", 16000, title="Generated music")
	#vis.mel_spectogram("D:\\MAESTRO\\maestro-v1.0.0\\2017\\MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--1.wav", 16000, title="Real music")
	plt.show()

