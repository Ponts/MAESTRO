import tensorflow as tf
from WaveNET import model as model
import genoptions as g
import discoptions as d
import loader
import numpy as np
from WaveNET import ops 
import progressbar
from time import sleep
import os, sys
import librosa
import time
import argparse
import matplotlib.pyplot as plt


logdir = "./tfb_logs/"
ablatelogs = "./ablate_logs"

def get_arguments():
	parser = argparse.ArgumentParser(description='WaveNet settings')
	parser.add_argument('--logdir', type=str, default=logdir,
		help='Which directory to save logs, restore model from, e.t.c.')

	return parser.parse_args()

def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
	reader = tf.train.NewCheckpointReader(save_file)
	saved_shapes = reader.get_variable_to_shape_map()
	var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
				if var.name.split(':')[0] in saved_shapes])    
	restore_vars = []
	for var_name, saved_var_name in var_names:
		curr_var = graph.get_tensor_by_name(var_name)
		var_shape = curr_var.get_shape().as_list()
		if var_shape == saved_shapes[saved_var_name]:
			restore_vars.append(curr_var)
	opt_saver = tf.train.Saver(restore_vars)
	opt_saver.restore(session, save_file)

def load(saver, sess, logDir):
	print("Trying to restore saved checkpoints from {} ...".format(logDir),
		end="")

	ckpt = tf.train.get_checkpoint_state(logDir)
	if ckpt:
		print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
		global_step = int(ckpt.model_checkpoint_path
						.split('/')[-1]
						.split('-')[-1])
		print("  Global step was: {}".format(global_step))
		print("  Restoring...", end="")
		saver.restore(sess, ckpt.model_checkpoint_path)
		print(" Done.")
		return global_step
	else:
		print(" No checkpoint found.")
		return None

def save(saver, sess, logdir, step):
	model_name = 'model.ckpt'
	checkpoint_path = os.path.join(logdir, model_name)
	print('Storing checkpoint to {} ...'.format(logdir), end="")
	sys.stdout.flush()

	if not os.path.exists(logdir):
		os.makedirs(logdir)

	saver.save(sess, checkpoint_path, global_step=step)
	print(' Done.')

def generate(length, conditionOn = None):
	filename="generated"
	sess = tf.Session()
	sr = g.options["sample_rate"]

	with tf.variable_scope("GEN/"):
		Generator = model.WaveNetModel(1,
			dilations=g.options["dilations"],
			filter_width=g.options["filter_width"],
			residual_channels=g.options["residual_channels"],
			dilation_channels=g.options["dilation_channels"],
			skip_channels=g.options["skip_channels"],
			quantization_channels=g.options["quantization_channels"],
			use_biases=g.options["use_biases"],
			scalar_input=g.options["scalar_input"],
			initial_filter_width=g.options["initial_filter_width"],
			#global_condition_channels=g.options["noise_dimensions"],
			global_condition_cardinality=None,
			histograms=True,
			add_noise=True)
	# Get the graph
	variables_to_restore = {
		var.name[:-2]: var for var in tf.global_variables()
		if not ('state_buffer' in var.name or 'pointer' in var.name) and "GEN/" in var.name}
	#print(len(variables_to_restore))

	saver = tf.train.Saver(variables_to_restore)
	print("Restoring model")
	ckpt = tf.train.get_checkpoint_state(logdir)
	saver.restore(sess, ckpt.model_checkpoint_path)
	print("Model {} restored".format(ckpt.model_checkpoint_path))

	sampleph = tf.placeholder(tf.float32, [1,Generator.receptive_field,1])
	noiseph = tf.placeholder(tf.float32, [1,1,g.options["noise_dimensions"]])
	encoded = ops.mu_law_encode(sampleph, g.options["quantization_channels"])
	sample = tf.placeholder(tf.float32)

	one_hot = Generator._one_hot(encoded)
	next_sample = Generator._create_network(one_hot, None, noise = noiseph)
	arg_maxes = tf.nn.softmax(next_sample, axis=2)
	decoded = ops.mu_law_decode(sample, g.options["quantization_channels"])
	#print(np.shape(arg_maxes))
	# Sampling with argmax atm
	#intermed = tf.sign(tf.reduce_max(arg_maxes, axis=2, keepdims=True)-arg_maxes)
	#one_hot = (intermed-1)*(-1)
	#fake_sample = tf.concat((tf.slice(encoded, [0,1,0], [-1,-1,-1]), appendph),1)

	generated = []
	if conditionOn is not None:
		audio, sr = librosa.load(conditionOn, g.options["sample_rate"], mono=True)
		start = np.random.randint(0,len(audio)-Generator.receptive_field)
		fakey = audio[start:start+Generator.receptive_field]
		audio_start = fakey
		#fakey = sess.run(audio)
		#generated = fakey.tolist()
	else:
		fakey = [0.0] * (Generator.receptive_field-1)
		fakey.append(np.random.uniform())
		audio_start=[]
	noise = np.random.normal(g.options["noise_mean"], g.options["noise_variance"], size=g.options["noise_dimensions"]).reshape(1,1,-1)

	# REMOVE THIS LATER
	noise = np.zeros((1,1,100))
	fakey = np.reshape(fakey, [1,-1,1])
	gen = sess.run(encoded, feed_dict={sampleph : fakey})
	generated = gen#[0,:,0].tolist()
	fakey = sess.run(one_hot, feed_dict={sampleph : fakey})
	print(np.shape(generated))
	bar = progressbar.ProgressBar(maxval=length, \
		widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	for i in range(length):
		prediction = sess.run(arg_maxes, feed_dict={one_hot : fakey, noiseph : noise})

		#fakey = sess.run(fake_sample, feed_dict={encoded : fakey, appendph : prediction})
		newest_sample = prediction[-1,-1,:]
		#Sample from newest_sample

		np.seterr(divide='ignore')
		scaled_prediction = np.log(newest_sample) / 0.9#args.temperature
		scaled_prediction = (scaled_prediction -
							np.logaddexp.reduce(scaled_prediction))
		scaled_prediction = np.exp(scaled_prediction)
		np.seterr(divide='warn')
		# Prediction distribution at temperature=1.0 should be unchanged after
		# scaling.
		#print(np.argmax(scaled_prediction))
		sample = np.random.choice(
			np.arange(g.options["quantization_channels"]), p=scaled_prediction)
		sample = np.argmax(scaled_prediction)
		generated = np.append(generated, np.reshape(sample,[1,1,1]), 1)
		fakey = sess.run(one_hot, feed_dict={encoded : generated[:,-Generator.receptive_field:,:]})
		bar.update(i+1)

	bar.finish()
	generated=np.reshape(generated,[-1])
	decoded = sess.run(ops.mu_law_decode(generated, g.options["quantization_channels"]))
	generated = np.array(decoded)
	librosa.output.write_wav("Generated/gangen.wav", generated, sr, norm=True)

	# Name is dilated stack or postprocessing, index is between 0 and 49 for dilated stack
	# 0 and 50 for dilated stack
def feature_max(layerName, layerIndex, unit_index = None):
	sess = tf.Session()
	sr = g.options["sample_rate"]

	with tf.variable_scope("GEN/"):
		Generator = model.WaveNetModel(1,
			dilations=g.options["dilations"],
			filter_width=g.options["filter_width"],
			residual_channels=g.options["residual_channels"],
			dilation_channels=g.options["dilation_channels"],
			skip_channels=g.options["skip_channels"],
			quantization_channels=g.options["quantization_channels"],
			use_biases=g.options["use_biases"],
			scalar_input=g.options["scalar_input"],
			initial_filter_width=g.options["initial_filter_width"],
			#global_condition_channels=g.options["noise_dimensions"],
			global_condition_cardinality=None,
			histograms=True,
			add_noise=True)
	variables_to_restore = {
		var.name[:-2]: var for var in tf.global_variables()
		if not ('state_buffer' in var.name or 'pointer' in var.name) and "GEN/" in var.name}
	#print(len(variables_to_restore))

	saver = tf.train.Saver(variables_to_restore)
	print("Restoring model")
	ckpt = tf.train.get_checkpoint_state(logdir)
	saver.restore(sess, ckpt.model_checkpoint_path)
	print("Model {} restored".format(ckpt.model_checkpoint_path))

	sampleph = tf.placeholder(tf.float32, [1,Generator.receptive_field,1])
	zeros = np.zeros((1,1,g.options["noise_dimensions"]))
	encoded = ops.mu_law_encode(sampleph, g.options["quantization_channels"])


	one_hot = Generator._one_hot(encoded)
	to_optimise = Generator._get_layer_activation(layerName, layerIndex, one_hot, None, noise = zeros)
	if unit_index is not None and unit_index < np.shape(to_optimise)[2]:
		to_optimise = to_optimise[:,:,unit_index];
		print(np.shape(to_optimise))
	gs = tf.gradients(to_optimise, one_hot)[0]
	
	#prob_dist = np.random.randint(0, g.options["quantization_channels"], size= (1,Generator.receptive_field)) # Start with random noise
	#prob_dist = np.ones((1,Generator.receptive_field, g.options["quantization_channels"])) / (g.options["quantization_channels"])
	prob_dist = softmax(np.random.random_sample((1, Generator.receptive_field, g.options["quantization_channels"])))
	length = 1000
	bar = progressbar.ProgressBar(maxval=length, \
		widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()

	for i in range(length):
		#inp = sampleFrom(prob_dist)
		grads = sess.run(gs, feed_dict = {one_hot : prob_dist})
		#print(np.shape(grads))
		prob_dist += 0.01*grads
		prob_dist = softmax(prob_dist)
		#prob_dist = prob_dist - np.min(prob_dist, axis=2, keepdims=True)
		#prob_dist /= np.sum(prob_dist, axis=2, keepdims=True)
		#print(np.shape(prob_dist))
		bar.update(i+1)
	bar.finish()
	
	print(prob_dist)
	prob_dist = softmax(prob_dist)
	max_path = np.reshape(np.argmax(prob_dist, axis=2),[-1])
	#generated = np.argmax(sampleFrom(prob_dist),axis=2)
	#print(np.shape(generated))
	#generated=np.reshape(generated,[-1])
	#decoded = sess.run(ops.mu_law_decode(generated, g.options["quantization_channels"]))
	#generated = np.array(decoded)
	import matplotlib.pyplot as plt
	plt.imshow(np.reshape(prob_dist,( g.options["quantization_channels"],Generator.receptive_field)))
	title = layerName + ", layer: " + str(layerIndex) 
	if unit_index is not None:
		title += ", channel : " +str(unit_index)
	plt.title(title)
	plt.show()



	#librosa.output.write_wav("Generated/visualize"+layerName + str(layerIndex)+".wav", np.array([float(x) for x in max_path]), sr, norm=True)

	
def sampleFrom(prob_dist):
	length = np.shape(prob_dist)[1]
	inp = np.zeros((1, length, g.options["quantization_channels"]))
	for i in range(length):
		sample = np.random.choice(
			np.arange(g.options["quantization_channels"]), p=prob_dist[0,i,:])
		#one = np.zeros((g.options["quantization_channels"]))
		#one[sample] = 1.0
		inp[0,i,sample] = 1.0
	return inp


def softmax(x, ax=2):
	ex = np.exp(x)
	return ex / np.sum(ex, axis=ax, keepdims=True)


def ablate(layerNames, layerIndexes):
	ablations = {}
	means = {}
	variations = {}
	counters = {}
	sum2 = {}
	coord = tf.train.Coordinator()
	sess = tf.Session()
	sr = g.options["sample_rate"]

	with tf.variable_scope("GEN/"):
		Generator = model.WaveNetModel(g.options["batch_size"],
			dilations=g.options["dilations"],
			filter_width=g.options["filter_width"],
			residual_channels=g.options["residual_channels"],
			dilation_channels=g.options["dilation_channels"],
			skip_channels=g.options["skip_channels"],
			quantization_channels=g.options["quantization_channels"],
			use_biases=g.options["use_biases"],
			scalar_input=g.options["scalar_input"],
			initial_filter_width=g.options["initial_filter_width"],
			global_condition_cardinality=None,
			histograms=False,
			add_noise=True)
	variables_to_restore = {
		var.name[:-2]: var for var in tf.global_variables()
		if not (('state_buffer' in var.name or 'pointer' in var.name) and "GEN/" in var.name) }
	
	#print(len(variables_to_restore))
	# Data reading
	l = loader.AudioReader("maestro-v1.0.0/2017", g.options["sample_rate"], Generator.receptive_field, coord, stepSize=1, sampleSize=g.options["sample_size"], silenceThreshold=0.1)
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	l.startThreads(sess)

	saver = tf.train.Saver(variables_to_restore)
	print("Restoring model")
	ckpt = tf.train.get_checkpoint_state(logdir)
	saver.restore(sess, ckpt.model_checkpoint_path)
	print("Model {} restored".format(ckpt.model_checkpoint_path))

	#sampleph = tf.placeholder(tf.float32, [1,Generator.receptive_field,1])
	deque = l.deque(g.options["batch_size"])
	zeros = np.zeros((1,1,g.options["noise_dimensions"]))
	encoded = ops.mu_law_encode(deque, g.options["quantization_channels"])
	one_hot = Generator._one_hot(encoded)

	to_save = {}
	# Create dicts
	for name in layerNames:
		ablations[name] = {}
		means[name] = {}
		variations[name] = {}
		counters[name] = {}
		sum2[name] = {}
		for i in layerIndexes:
			ablations[name][i] = tf.reduce_mean(Generator._get_layer_activation(name, i, one_hot, None, noise=zeros), axis=[0,1]) 
			s2 = tf.Variable(tf.zeros(tf.shape(ablations[name][i])), name="ABL/zero_"+name+str(i))
			sum2[name][i] = s2.assign_add(ablations[name][i]**2)
			to_save["ABL/zero_"+name+str(i)] = s2

			c = tf.Variable(0,name="ABL/counter_"+name+str(i),dtype=tf.float32)
			counters[name][i] = c.assign_add(1)
			to_save["ABL/counter_"+name+str(i)] = c

			m = tf.Variable(tf.zeros(tf.shape(ablations[name][i])), name="ABL/mean_"+name+str(i))
			means[name][i] = m.assign(((m * c) + ablations[name][i] ) / (c+1))
			to_save["ABL/mean_"+name+str(i)] = m

			v = tf.Variable(tf.zeros(tf.shape(ablations[name][i])), name="ABL/var_"+name+str(i))
			variations[name][i] = v.assign(tf.sqrt((s2 / (c) ) - (m**2)))
			to_save["ABL/var_"+name+str(i)] = v



	sess.run(tf.global_variables_initializer())
	print("Dict created")
	print("Restoring previous statistics")
	ablatesaver = tf.train.Saver(to_save)
	ablateckpt = tf.train.get_checkpoint_state(ablatelogs)
	if ablateckpt is not None:
		optimistic_restore(sess, ablateckpt.model_checkpoint_path, tf.get_default_graph())
	print("Statistics restored")

	ms = []
	# Gather statistics
	for k in range(300):
		for name in layerNames:
			for i in layerIndexes:
				print(k)
				sess.run(sum2[name][i])
				ms.append(sess.run(means[name][i])[0])
				sess.run(variations[name][i])
				c = sess.run(counters[name][i])


	#mean = sess.run(means['dilated_stack'][3])
	#print(np.shape(mean))

	plt.plot(ms)
	plt.show()

	model_name = 'ablate.ckpt'
	checkpoint_path = os.path.join(ablatelogs, model_name)
	ablatesaver.save(sess, checkpoint_path)

	coord.request_stop()
	coord.join(threads)







def train(coord, G, D, loader, fw):
	# Get the graph
	conf = tf.ConfigProto(log_device_placement=False)
	conf.gpu_options.allow_growth=True
	sess = tf.Session(config=conf)
	init = tf.global_variables_initializer()
	sess.run(init)
	summaries = tf.summary.merge_all()
	# Saver
	saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=5)
	try:
		last_global_step = load(saver, sess, logdir)
		if last_global_step is not None:
			print("Restored model")
	except:
		print("Error in restoring model, terminating")
		raise
	
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	loader.startThreads(sess)

	
	# Generate samples of the Discriminators receptive field
	samples = []
	recField = D.receptive_field
	print("Receptive field: " + str(recField))
	#sample = sess.run(loader.deque(g.options["batch_size"]))
	#print(loader.deque(g.options["batch_size"]))
	#for i in range(recField):
	#sam = sess.run(fake_sample)
	step = last_global_step
	last_saved_step = last_global_step
	genPretrainingSteps = 1600000
	discPretrainingSteps = 20000
	if last_saved_step is None:
		last_saved_step = 0
		step = 0
	try:			   
		for j in range(2000000):
			if step < genPretrainingSteps: # Do pretraining training
				startTime = time.time()
				_, lossMl = sess.run([mlStep, mlLoss])
				'''
				if step % 10000 == 0: and step > 10000:
					init_noise = sess.run(noise)
					init_audio = sess.run(audio)
					bar = progressbar.ProgressBar(maxval=recField, \
						widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
					bar.start()
					for k in range(recField):
						fakey = sess.run(fake_sample, feed_dict={audio:init_audio, noise:init_noise})
						#print(fakey[-1,-1,-1])
						bar.update(k+1)
					_, dLoss = sess.run([discStep, discLoss], feed_dict={fake_sample : fakey, noise : init_noise, daudio : init_audio})
					bar.finish()
					#_, dLoss = sess.run([discStep, discLoss])
					#base, gen, target = sess.run([mldequed, ml_one_step, mltarget])
					print("Disc Loss : " + str(dLoss))
				'''
				dur = time.time() - startTime
				print("MLLoss GEN: " + str(lossMl) + ", step: " + str(step) + ", {:.3f} secs".format(dur))
			elif step < genPretrainingSteps + discPretrainingSteps:
				startTime = time.time()
				_, dLoss = sess.run([discStep, discLoss]); 
				dur = time.time() - startTime
				print("Disc pre Loss :  " + str(dLoss) + ", Time: " + str(dur))
				
			else: # Do gan-training
				save(saver, sess, logdir, step)
				return
				startTime = time.time()

				# Train D 5 times	
				for i in range(5):
					#print("Getting audio")
					init_noise = sess.run(noise)
					init_audio = sess.run(audio)
					#sess.run(discStep, feed_dict={codedNoise : init_noise})
					# This row should be replaced with longer samples later in training? now done
					#bar = progressbar.ProgressBar(maxval=recField, \
					#	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
					#bar.start()
					#for k in range(recField):
					fakey = sess.run(fake_sample, feed_dict={audio:init_audio, noise:init_noise})
					#print(fakey[-1,-1,-1])
					#	bar.update(k+1)
					sess.run(discStep, feed_dict={fake_sample : fakey, noise : init_noise, daudio : init_audio})
					#bar.finish()
					#print("Done")

				# Train G 1 time
				#_, dLoss, gLoss = sess.run([genStep, discLoss, genLoss], feed_dict={audio : init_audio, codedNoise : init_noise, daudio : init_audio})
				#fakey = sess.run(audio)
				#init_noise = sess.run(noise)
				#for k in range(recField):
				#fakey = sess.run(fake_sample, feed_dict={audio : fakey, noise : init_noise})
				#_, dLoss, gLoss, slopeLoss, gradsLoss = sess.run([genStep, discLoss, genLoss, nr, slope], feed_dict={fake_sample : fakey, noise : init_noise})
				_, dLoss, gLoss = sess.run([genStep, discLoss, genLoss])
				dur = time.time() - startTime
				print("DiscLoss:  " + str(dLoss))
				#print("SlopeLoss: " + str(slopeLoss))
				#print("GradsLoss: " + str(gradsLoss))
				print("GenLoss:   " + str(gLoss) + ", Step: " + str(step) + ", Time: " + str(dur))
				print()
			step += 1
			if step % 100 == 0:
				summs = sess.run(summaries)
				fw.add_summary(summs,step)
			if step % 500 == 0:
				save(saver, sess, logdir, step)
				


	except KeyboardInterrupt:
		# Introduce a line break after ^C is displayed so save message
		# is on its own line.
		print()
	finally:
		if step > last_saved_step:
			save(saver, sess, logdir, step)
			coord.request_stop()
			coord.join(threads)
	


if __name__ == "__main__":
	args = get_arguments()
	logdir = args.logdir
	modes = ["Generate", "FeatureVis", "Train", "Ablate"]
	mode = modes[0]

	if mode == modes[0]: #Generate
		generate(16000*1, "D:\\MAESTRO\\maestro-v1.0.0\\2017\\MIDI-Unprocessed_051_PIANO051_MID--AUDIO-split_07-06-17_Piano-e_3-02_wav--2.wav")
	elif mode == modes[1]: # FeatureVis
		feature_max('postprocessing', 0, None)
	elif mode == modes[3]: #ABLATE
		ablate(['dilated_stack'], [3]);
	elif mode == modes[2]: #TRAIN
		fw = tf.summary.FileWriter(logdir)
		coord = tf.train.Coordinator()

		with tf.variable_scope("GEN/"):
			Generator = model.WaveNetModel(g.options["batch_size"],
				dilations=g.options["dilations"],
				filter_width=g.options["filter_width"],
				residual_channels=g.options["residual_channels"],
				dilation_channels=g.options["dilation_channels"],
				skip_channels=g.options["skip_channels"],
				quantization_channels=g.options["quantization_channels"],
				use_biases=g.options["use_biases"],
				scalar_input=g.options["scalar_input"],
				initial_filter_width=g.options["initial_filter_width"],
				#global_condition_channels=g.options["noise_dimensions"],
				global_condition_cardinality=None,
				histograms=True,
				final_layer_size=g.options["quantization_channels"],
				add_noise=True)


		with tf.variable_scope("DIS/"):
			Discriminator = model.WaveNetModel(g.options["batch_size"],
				dilations=d.options["dilations"],
				filter_width=d.options["filter_width"],
				residual_channels=d.options["residual_channels"],
				dilation_channels=d.options["dilation_channels"],
				skip_channels=d.options["skip_channels"],
				quantization_channels=d.options["quantization_channels"],
				use_biases=d.options["use_biases"],
				scalar_input=d.options["scalar_input"],
				initial_filter_width=d.options["initial_filter_width"],
				global_condition_channels=None,
				global_condition_cardinality=None,
				histograms = True,
				final_layer_size=d.options["final_layer_size"]) # Disc should output 1
		# Audio dimensions: [Song, Samplenr, Quantization]
		# Data loading
		channels = g.options["quantization_channels"]
		with tf.name_scope("Pre-processing"):
			l = loader.AudioReader("maestro-v1.0.0/2017", g.options["sample_rate"], Discriminator.receptive_field, coord, stepSize=1, sampleSize=g.options["sample_size"], silenceThreshold=0.1)
			abatch = l.deque(g.options["batch_size"])
			mldequed = l.dequeMl(g.options["batch_size"])
			# Quantize audio into channels
			mldequed = ops.mu_law_encode(mldequed, channels)
			mldequed = Generator._one_hot(mldequed)
			mldequed = tf.reshape(tf.cast(mldequed, tf.float32), [g.options["batch_size"], -1, channels])
			mltarget = tf.slice(mldequed,[0,Generator.receptive_field,0],[-1,-1,-1])
			mlinput = tf.slice(mldequed,[0,0,0],[-1, tf.shape(mldequed)[1]-1, -1])
			#Quantize encoded into channels
			abatch = ops.mu_law_encode(abatch, channels)
			encoded = Generator._one_hot(abatch)
			#encoded = tf.reshape(tf.cast(abatch, tf.float32), [g.options["batch_size"], -1, channels])

			noise = l.dequeNoise(g.options["batch_size"])

		# Generator stuff
		with tf.variable_scope("GEN/", reuse=tf.AUTO_REUSE):
			audio = encoded
			#noise = Generator._embed_gc(codedNoise)
			zeros = tf.zeros(tf.shape(noise),tf.float32)
			one_step = Generator._create_network(audio, None, noise=noise)
		with tf.variable_scope("GEN/", reuse=True):
			ml_one_step = Generator._create_network(mlinput, None, noise=zeros) # Might be dragons
			with tf.name_scope("Generating/"):
				# Get sample by argmax for now
				#gs = tf.gradients(one_step, audio)[0]
				#print(np.shape(gs))
				softies = tf.nn.softmax(one_step, axis=2)
				#print(np.shape(arg_maxes))
				#arg_maxes = tf.sign(tf.reduce_max(softies, axis=2, keepdims=True)-softies)
				#arg_maxes = (arg_maxes-1)*(-1)
				#print(np.shape(arg_maxes))
				#newest_sample = softies[:,-1,:]
				#Sample from newest_sample

				scaled_prediction = tf.log(softies) / 0.9 #args.temperature
				loged = tf.log(tf.reduce_sum(tf.exp(scaled_prediction), axis=2, keepdims = True))
				#print(np.shape(loged))
				scaled_prediction = (scaled_prediction - loged)
				#print(np.shape(scaled_prediction))
				#scaled_prediction = tf.exp(scaled_prediction)
				mask_indexes = tf.multinomial(tf.reshape(scaled_prediction[:,0,:], [g.options["batch_size"], g.options["quantization_channels"]]), 1)
				#print(np.shape(mask_indexes))
				#mask_index = np.random.choice(
				#		np.arange(g.options["quantization_channels"]), p=scaled_prediction)
				
				mask = Generator._one_hot(mask_indexes)
				arg_maxes = tf.sign(softies * mask)

				#GS = tf.gradients(arg_maxes, softies)[0]
				#print("GRADS")
				#print(np.shape(GS))
				#print(np.shape(arg_maxes))
				#arg_maxes = tf.sign(arg_maxes )
				#print(np.shape(arg_maxes))
				#gs = tf.gradients(arg_maxes, audio)[0]
				#print(np.shape(gs))
				#one_step = Generator._one_hot(arg_maxes)

				#print(np.shape(one_step))
				# Shift the generated vector
				fake_sample = tf.concat((tf.slice(audio, [0,1,0], [-1,-1,-1]), arg_maxes),1)
				#print("fake sample")
				#print(np.shape(fake_sample))

		
		# Discriminator stuff
		with tf.variable_scope("DIS/", reuse=tf.AUTO_REUSE):
			daudio = encoded
			print("Diff in gen and disc receptive field:")
			print(tf.shape(daudio)[1]-Discriminator.receptive_field)
			# Make sure audio samples is of the correct length
			daudio = tf.slice(daudio, [0,tf.shape(daudio)[1]-Discriminator.receptive_field,0], [-1,-1,-1])
			r_logits = Discriminator._create_network(daudio, None)
			#r_logits = tf.layers.dense(r_logits,1, name="Final_Layer")
			#r_logits = tf.layers.dense(real_output, 1, name="final_D")
		with tf.variable_scope("DIS/", reuse=True):
			fake_sample = tf.slice(fake_sample, [0,tf.shape(daudio)[1]-Discriminator.receptive_field,0], [-1,-1,-1])
			f_logits = Discriminator._create_network(fake_sample, None)
			#f_logits = tf.layers.dense(f_logits,1, name="Final_Layer")
			#f_logits = tf.layers.dense(fake_output, 1, name="final_D")

		# Get the variables
		genVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GEN/")
		discVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DIS/")

		with tf.name_scope("GP/"):
			# GP
			e = tf.random_uniform([g.options["batch_size"]],0,1) #WRONG
			e = tf.reshape(e, (g.options["batch_size"],-1,1))
			e = tf.tile(e, [1,tf.shape(daudio)[1],channels])
			polated = tf.add(tf.multiply(daudio,e), tf.multiply(fake_sample,1-e))
		print("shape polated")
		print(np.shape(polated))

		with tf.variable_scope("DIS/", reuse=True):
			discPolated = Discriminator._create_network(polated, None)
			
		with tf.name_scope("GP/"):
			grads = tf.gradients(discPolated,polated)[0]
			slope = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1,2]))
			nr = 10*tf.reduce_mean((slope-1)**2)


		with tf.name_scope("Loss/"):
			discLoss = -tf.reduce_mean(r_logits) + tf.reduce_mean(f_logits) + 10*tf.reduce_mean((slope-1)**2)
			genLoss = -tf.reduce_mean(f_logits)
			mlLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=ml_one_step, labels=mltarget))

		tf.summary.scalar('Discriminator Loss',discLoss)
		tf.summary.scalar('Generator Loss', genLoss)
		tf.summary.scalar('ML Loss', mlLoss)

		#gs = tf.gradients(f_logits, audio)[0]
		#print()
		#print(np.shape(gs))
		#print(np.shape(f_logits))
		#print(np.shape(r_logits))

		mlStep = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0, beta2=0.9).minimize(mlLoss, var_list=genVars)
		genStep = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0, beta2=0.9).minimize(genLoss, var_list=genVars)
		discStep = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0, beta2=0.9).minimize(discLoss, var_list=discVars)
		
		graph = tf.get_default_graph()
		fw.add_graph(graph)

		train(coord, Generator, Discriminator, l, fw)

