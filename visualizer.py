import librosa, librosa.display
import fnmatch
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pyaudio 
import wave
import math
from scipy.signal import blackmanharris, fftconvolve
from scipy.spatial.distance import directed_hausdorff, euclidean
from matplotlib.mlab import find
import similaritymeasures
from fastdtw import fastdtw
from dtw import dtw

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

	def getFreq(self, note):
		for k, n in self.frequencies.items():
			if n == note:
				return k
		return -1

	def loadAudio(self, file, sar):
		audio, sr = librosa.load(file, sar, mono = True)
		return audio, sr

	# https://github.com/endolith/waveform-analyzer/blob/master/frequency_estimator.py
	def parabolic(self, f, x): 
		xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
		yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
		return (xv, yv)
	    
	# https://github.com/endolith/waveform-analyzer/blob/master/frequency_estimator.py
	def freq_from_autocorr(self, raw_data_signal, fs):                          
		corr = fftconvolve(raw_data_signal, raw_data_signal[::-1], mode='full')
		corr = corr[int(len(corr)/2):]
		d = np.diff(corr)
		start = find(d > 0)[0]
		peak = np.argmax(corr[start:]) + start
		px, py = self.parabolic(corr, peak)
		return fs / px

	def loudness(self, chunk):
		data = np.array(chunk, dtype=float) / 32768.0
		ms = math.sqrt(np.sum(data ** 2.0) / len(data))
		if ms < 10e-8: 
			ms = 10e-8
		return 10.0 * math.log(ms, 10.0)

	def detectNote(self, audio, sr):
		freq = self.freq_from_autocorr(audio, sr)
		bestDist = freq
		for k, _ in self.frequencies.items():
			distance = np.abs(freq - k)
			if distance < bestDist:
				bestDist = distance
				bestFreq = k
		return self.frequencies[bestFreq]
		'''
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
		'''


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
		prevNote = ""
		while data != '':
			# writing to the stream is what *actually* plays the sound.
			stream.write(data)
			data = w.readframes(sl)
			if i+sl >= len(audio):
				break
			note = self.detectNote(audio[i:i+sl], sr) # Move this to the detect note function
			loudness = np.abs(self.loudness(audio[i:i+sl]))
			if loudness < 60.:
				print("note: %s, amp %0.4f"%(note, loudness))
				prevNote = note
			i += 1

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
			#ra = np.random.randint(0,len(audio)-sr)
			audio = audio[-sar:]
		#print(len(audio))
		#print(sr)
		S = librosa.feature.melspectrogram(y=audio, sr=sr)
		fig = plt.figure(figsize=(10, 4))
		librosa.display.specshow(librosa.power_to_db(S,
													ref=np.max),
													y_axis='mel', fmax=sr/2,
													x_axis='time')
		plt.colorbar(format='%+2.0f dB')
		plt.title(title)
		plt.tight_layout()
		
	def check_similarity(self, song1, option1, option2, sar=16000):
		song1, sr1 = self.loadAudio(song1, sar)
		opt1, sr2 = self.loadAudio(option1, sar)
		opt2, sr3 = self.loadAudio(option2, sar)
		x = range(0,5117)
		song1 = [x, song1[-5117:]]
		opt1 = [x, opt1[-5117:]]
		opt2 = [x, opt2[-5117:]]

		nsong1 = np.array(song1).T
		nopt1 = np.array(opt1).T
		nopt2 = np.array(opt2).T

		print(np.shape(nopt1))

		dtw1, d = similaritymeasures.dtw(opt1, song1)
		dtw2, d = similaritymeasures.dtw(opt2, song1)
		#print("DTW")
		#print(dtw1, dtw2)

		#dtw1, d = fastdtw(nopt1, nsong1, dist=euclidean)
		#dtw2, d = fastdtw(nopt2, nsong1, dist=euclidean)
		#print("fastDTW")
		#print(dtw1, dtw2)
		#dtw1 = similaritymeasures.frechet_dist(nopt1, nsong1)
		#dtw2 = similaritymeasures.frechet_dist(nopt2, nsong1)
		#print("fastDTW")
		#print(dtw1, dtw2)

		#dh1, _, _ = directed_hausdorff(nopt1, nsong1)
		#dh2, _, _ = directed_hausdorff(nopt2, nsong1)
		#print("Directed Hausdorff")
		#print(dh1, dh2)

		#dtw1 = similaritymeasures.pcm(nopt1, nsong1)
		#dtw2 = similaritymeasures.pcm(nopt2, nsong1)
	
		print(dtw1, dtw2)

		return dtw1, dtw2

	def compare_frechet(self):
		x = np.arange(0, 10*np.pi, np.pi/10)
		s1 = np.cos(x)
		s2 = np.cos(x + np.pi/2)


		plt.plot(s1)
		plt.plot(s2)
		plt.show()

		s1 = np.array([x, s1]).T
		s2 = np.array([x, s2]).T
		dist,_ = similaritymeasures.dtw(s1, s2)


		print(dist)

	def get_scores(self):
		uncons = []
		cons = []
		for index in range(1,100):
			uncon, con = vis.check_similarity("D:\\MAESTRO\\Generated\\Comparision\\to_copy_"+str(index)+".wav", "D:\\MAESTRO\\Generated\\Comparision\\uncontrolled_"+str(index)+".wav","D:\\MAESTRO\\Generated\\Comparision\\controlled_"+str(index)+".wav")
			uncons.append(uncon)
			cons.append(con)
		print("Controlled score:")
		print(str(np.mean(cons)) + "+-" + str(np.std(cons)))
		print("Not Controlled score:")
		print(str(np.mean(uncons)) + "+-" + str(np.std(uncons)))

	def plotthat(self):
		con = [19.69274514118228, 22.668568919557394, ]
		uncon = [22.359309147870928, 18.86512833495965, ]



if __name__ == "__main__":
	vis = Visualizer(2**16)
	#vis.visAudio("D:\\MAESTRO\\maestro-v1.0.0\\2017\\MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--2.wav", dynamicSl = True, time = 0.02)
	#vis.visAudio("D:\\MAESTRO\\Generated\\gangen.wav", dynamicSl = True, time=0.02)
	#vis.visAudio("D:\\MAESTRO\\Generated\\visualizedilated_stack5.wav", dynamicSl = True, time=0.02)
	#vis.visAudio("D:\\normal_wavenet\\generate.wav", dynamicSl = True, time=0.02)
	#vis.testDetector("D:\\MAESTRO\\maestro-v1.0.0\\2017\\MIDI-Unprocessed_047_PIANO047_MID--AUDIO-split_07-06-17_Piano-e_2-04_wav--4.wav", time = 0.05)
	#vis.compare("D:\\MAESTRO\\Generated\\gangen.wav", "D:\\MAESTRO\\Generated\\bla.wav")
	#vis.testDetector("D:\\MAESTRO\\Generated\\gangen2.wav", time = 0.2)
	#vis.mel_spectogram("D:\\MAESTRO\\Generated\\investigate.wav", 16000, title="Generated music")
	#vis.mel_spectogram("D:\\MAESTRO\\Generated\\to_copy.wav", 16000, title="Real music")
	
	#index = 
	'''
	for index in range(1, 75):
		vis.mel_spectogram("D:\\MAESTRO\\Generated\\Comparision\\controlled_"+str(index)+".wav", 16000, title="")
		plt.savefig('C:\\Users\\pontu\\OneDrive\\Bilder\\Thesis Results\\'+str(index) + '_c', bbox_inches='tight')
		plt.close()
		vis.mel_spectogram("D:\\MAESTRO\\Generated\\Comparision\\uncontrolled_"+str(index)+".wav", 16000, title="")
		plt.savefig('C:\\Users\\pontu\\OneDrive\\Bilder\\Thesis Results\\'+str(index) + '_nc', bbox_inches='tight')
		plt.close()
		vis.mel_spectogram("D:\\MAESTRO\\Generated\\Comparision\\to_copy_"+str(index)+".wav", 16000, title="")
		plt.savefig('C:\\Users\\pontu\\OneDrive\\Bilder\\Thesis Results\\'+str(index) + '_copy', bbox_inches='tight')
		plt.close()
	'''
	#plt.show()
	con = []
	uncon = []
	for index in range(200,212):
		
		u, c= vis.check_similarity("D:\\MAESTRO\\Generated\\Comparision\\to_copy_"+str(index)+".wav", "D:\\MAESTRO\\Generated\\Comparision\\uncontrolled_"+str(index)+".wav","D:\\MAESTRO\\Generated\\Comparision\\controlled_"+str(index)+".wav")
		con.append(c)
		uncon.append(u)

	xs = [0]
	for i in range(len(con)-1):
		xs.append(xs[i] + 5117.0/16000.0)

	plt.plot(xs, con, label="Controlled")
	plt.plot(xs, uncon, label="Not controlled")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("DTW")
	plt.show()


	#vis.get_scores()
	#vis.compare_frechet()

	#vis.mel_spectogram("D:\\MAESTRO\\Generated\\c3#controlledcausalifALLuselatestactlayer29.wav", 16000, title="")
	#plt.show()