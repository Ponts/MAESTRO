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
import visualizer as visul


logdir = "./tfb_logs/"
ablatelogs = "./ablate_logs"
#generatedir="./firstModel/0511ganlosslow--1.4"
generatedir = "./tfb_logs/"
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
	#ckpt = tf.train.latest_checkpoint(logDir)
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
	ckpt = tf.train.get_checkpoint_state(generatedir)
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
		#audio_start = fakey
		#fakey = sess.run(audio)
		#generated = fakey.tolist()
	else:
		fakey = [0.0] * (Generator.receptive_field-1)
		fakey.append(np.random.uniform())
		#audio_start=[]
	noise = np.random.normal(g.options["noise_mean"], g.options["noise_variance"], size=g.options["noise_dimensions"]).reshape(1,1,-1)

	# REMOVE THIS LATER
	#noise = np.zeros((1,1,100))
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
		#print(newest_sample)
		#np.seterr(divide='ignore')
		#scaled_prediction = np.log(newest_sample) / 0.9#args.temperature
		#scaled_prediction = (scaled_prediction -
		#					np.logaddexp.reduce(scaled_prediction))
		#scaled_prediction = np.exp(scaled_prediction)
		#np.seterr(divide='warn')
		#print(np.sum(newest_sample - scaled_prediction))
		# Prediction distribution at temperature=1.0 should be unchanged after
		# scaling.
		#print(np.argmax(scaled_prediction))

		scaled_prediction = newest_sample

		sample = np.random.choice(
			np.arange(g.options["quantization_channels"]), p=scaled_prediction)
		#sample = np.argmax(newest_sample)
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
		
	print("to_optimise shape")
	print(np.shape(to_optimise))
	gs = tf.gradients(to_optimise, one_hot)[0]
	
	#prob_dist = np.random.randint(0, g.options["quantization_channels"], size= (1,Generator.receptive_field)) # Start with random noise
	#prob_dist = np.ones((1,Generator.receptive_field, g.options["quantization_channels"])) / (g.options["quantization_channels"])
	prob_dist = softmax(np.random.random_sample((1, Generator.receptive_field, g.options["quantization_channels"])))
	length = 2048
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


def investigate(layerNames, layerIndexes, conditionOn):
	vis = visul.Visualizer(2*16)
	means = {}
	variations = {}
	ablations = {}
	coord = tf.train.Coordinator()
	sess = tf.Session()
	to_restore = {}
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
			global_condition_cardinality=None,
			histograms=False,
			add_noise=True)
	variables_to_restore = {
		var.name[:-2]: var for var in tf.global_variables()
		if not (('state_buffer' in var.name or 'pointer' in var.name) and "GEN/" in var.name) }

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

	
	
	audio, sr = librosa.load(conditionOn, g.options["sample_rate"], mono=True)
	start = np.random.randint(0,len(audio)-Generator.receptive_field)
	fakey = audio[start:start+Generator.receptive_field]
	noise = np.random.normal(g.options["noise_mean"], g.options["noise_variance"], size=g.options["noise_dimensions"]).reshape(1,1,-1)

	for name in layerNames:
		means[name] = {}
		ablations[name] = {}
		variations[name] = {}
		for i in layerIndexes:
			#ablations[name][i] = get_causal_activations(Generator._get_layer_activation(name, i, one_hot, None, noise=zeros),i)
			ablations[name][i] = Generator._get_layer_activation(name, i, one_hot, None, noise=noiseph)
			abl = tf.reduce_mean(ablations[name][i], axis=[0,1])
			means[name][i] = tf.Variable(tf.zeros(tf.shape(abl)), name="ABL/mean_"+name+str(i))
			to_restore["ABL/mean_"+name+str(i)] = means[name][i]
			variations[name][i] = tf.Variable(tf.zeros(tf.shape(abl)), name="ABL/var_"+name+str(i))
			to_restore["ABL/var_"+name+str(i)] = variations[name][i]


	print("Restoring previous statistics")
	ablatesaver = tf.train.Saver(to_restore)
	ablateckpt = tf.train.get_checkpoint_state(ablatelogs)
	if ablateckpt is not None:
		optimistic_restore(sess, ablateckpt.model_checkpoint_path, tf.get_default_graph())
	print("Statistics restored")

	name = layerNames[0]
	i = layerIndexes[0]
	
	limits = means[name][i] + variations[name][i]
	mask = ablations[name][i] > limits

	
	fakey = np.reshape(fakey, [1,-1,1])
	generated = sess.run(encoded, feed_dict={sampleph : fakey})
	fakey = sess.run(one_hot, feed_dict={sampleph : fakey})
	sl = 100
	time = sl/g.options["sample_rate"]
	print(time)
	length=1000
	bar = progressbar.ProgressBar(maxval=length, \
		widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	prevNote = ""
	counter = 0
	for k in range(length):
		act, prediction = sess.run([ablations[name][i], arg_maxes], feed_dict={one_hot : fakey, noiseph : noise})
		#fakey = sess.run(fake_sample, feed_dict={encoded : fakey, appendph : prediction})
		newest_sample = prediction[-1,-1,:]

		sample = np.random.choice(
			np.arange(g.options["quantization_channels"]), p=newest_sample)
		#sample = np.argmax(newest_sample)			
		generated = np.append(generated, np.reshape(sample,[1,1,1]), 1)
		if counter > sl:
			counter = 0
			decoded = sess.run(ops.mu_law_decode(generated[0,-sl:,0], g.options["quantization_channels"]))
			note, amp = vis.detectNote(decoded, time)
			print("note: %s, amp %0.4f"%(note, amp))
			if prevNote != note and amp > 1.:
				print("note: %s, amp %0.4f"%(note, amp))
				prevNote = note
		counter += 1
		fakey = sess.run(one_hot, feed_dict={encoded : generated[:,-Generator.receptive_field:,:]})
		bar.update(k+1)


def get_causal_activations(activations, layerIndex):
	jump = g.options["dilations"][layerIndex + 1]
	# shape is batch, time, channel
	# want to get correct time ones
	end = tf.shape(activations)[1] #- 1
	indices = tf.range(end, -1, -jump)
	return tf.gather(activations, indices, axis=1)

	
def create_histograms(layerNames, layerIndexes):
	activations = {}
	summaries = []
	coord = tf.train.Coordinator()
	sess = tf.Session()
	writer = tf.summary.FileWriter("histograms")

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

	for name in layerNames:
		activations[name] = {}
		for i in layerIndexes:
			#activations[name][i] = get_causal_activations(Generator._get_layer_activation(name, i, one_hot, None, noise=zeros), i)
			activations[name][i] = Generator._get_layer_activation(name, i, one_hot, None, noise=zeros)
			for l in range(g.options["residual_channels"]):
				tf.summary.histogram(name + "_layer_" + str(i) + "_unit_" + str(0), tf.reshape(activations[name][i][:,:,l], (-1,1)))
			#units = tf.shape(activations[name][i])[2]
			#def add(a):
			#	tf.summary.histogram(name + "_layer_" + str(i) + "_unit_" + str(a), activations[name][i][:,:,a])
			#	tf.add(a,1)
			#	return tf.constant(0)

			#stop = lambda a: tf.less(a, units)

			#tf.while_loop(stop, add, [0])
				
	summaries = tf.summary.merge_all()
	#acts = []
	for i in range(1000):
		summ = sess.run(summaries)
		#act = sess.run(activations['dilated_stack'][0])[0,:,0]
		#print(act)
		print(i/1000)
		#acts = np.concatenate((acts, act))
		writer.add_summary(summ, global_step = 0)

	#plt.hist(acts, 32)
	#plt.show()

	
	coord.request_stop()
	coord.join(threads)


def ablate(layerNames, layerIndexes):
	sm = {}
	activations = {}
	means = {}
	variations = {}
	counters = {}
	sum2 = {}
	batch_size = {}
	sum2save = {}
	sum2saveop = {}
	meanssaveop = {}
	counterssaveop = {}
	variationssaveop = {}
	coord = tf.train.Coordinator()
	sess = tf.Session()

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
	
	# Data reading
	l = loader.AudioReader("maestro-v1.0.0/2017", g.options["sample_rate"], Generator.receptive_field, coord, stepSize=1, sampleSize=g.options["sample_size"], silenceThreshold=0.1)
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	l.startThreads(sess)

	saver = tf.train.Saver(variables_to_restore)
	print("Restoring model")
	ckpt = tf.train.get_checkpoint_state(logdir)
	saver.restore(sess, ckpt.model_checkpoint_path)
	print("Model {} restored".format(ckpt.model_checkpoint_path))

	deque = l.deque(g.options["batch_size"])
	zeros = np.zeros((1,1,g.options["noise_dimensions"]))
	encoded = ops.mu_law_encode(deque, g.options["quantization_channels"])
	one_hot = Generator._one_hot(encoded)

	to_save = {}
	# Create dicts
	for name in layerNames:
		sm[name] = {}
		activations[name] = {}
		means[name] = {}
		variations[name] = {}
		counters[name] = {}
		sum2[name] = {}
		batch_size[name] = {}
		sum2save[name] = {}
		sum2saveop[name] = {}
		meanssaveop[name] = {}
		counterssaveop[name] = {}
		variationssaveop[name] = {}
		for i in layerIndexes:
			#activations[name][i] = get_causal_activations(Generator._get_layer_activation(name, i, one_hot, None, noise=zeros), i)
			activations[name][i] = Generator._get_layer_activation(name, i, one_hot, None, noise=zeros)
			sm[name][i] = tf.reduce_sum(activations[name][i], axis=[0,1])
			sum2[name][i] = tf.reduce_sum(tf.square(activations[name][i]), axis=[0,1])		
			batch_size[name][i] = tf.to_float(tf.shape(activations[name][i])[0] + tf.shape(activations[name][i])[1])


			# Save variables
			sum2save[name][i] = tf.Variable(tf.zeros(tf.shape(sm[name][i])), name="ABL/sum2_"+name+str(i))
			to_save["ABL/sum2_"+name+str(i)] = sum2save[name][i]

			counters[name][i] = tf.Variable(0,name="ABL/counter_"+name+str(i),dtype=tf.float32)
			to_save["ABL/counter_"+name+str(i)] = counters[name][i]

			means[name][i] = tf.Variable(tf.zeros(tf.shape(sm[name][i])), name="ABL/mean_"+name+str(i))
			to_save["ABL/mean_"+name+str(i)] = means[name][i]

			variations[name][i] = tf.Variable(tf.zeros(tf.shape(sm[name][i])), name="ABL/var_"+name+str(i))
			to_save["ABL/var_"+name+str(i)] = variations[name][i]
			
			sum2saveop[name][i] = tf.assign(sum2save[name][i], sum2save[name][i] + sm[name][i])
			meanssaveop[name][i] = tf.assign(means[name][i], ((means[name][i] * counters[name][i]) + sm[name][i]) / (counters[name][i] + batch_size[name][i] )  )
			counterssaveop[name][i] = tf.assign(counters[name][i], counters[name][i] + batch_size[name][i])
			variationssaveop[name][i] = tf.assign(variations[name][i], tf.sqrt(tf.abs((sum2save[name][i] / counters[name][i]) - tf.square(means[name][i]))))

	sess.run(tf.global_variables_initializer())
	
	print("Dict created")
	print("Restoring previous statistics")
	ablatesaver = tf.train.Saver(to_save)
	ablateckpt = tf.train.get_checkpoint_state(ablatelogs)
	if ablateckpt is not None:
		optimistic_restore(sess, ablateckpt.model_checkpoint_path, tf.get_default_graph())
	print("Statistics restored")
	# Eat up some so that statistics arent gathered at the beginning
	for _ in range(1000):
		sess.run(deque)
	# Gather statistics
	# How much statistics do we need? Preferably a lot :)
	length = 10000
	bar = progressbar.ProgressBar(maxval=length, \
		widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	for k in range(length):
		for name in layerNames:
			for i in layerIndexes:
				act = sess.run(activations[name][i])
				sess.run([sum2saveop[name][i], meanssaveop[name][i], counterssaveop[name][i]], feed_dict={activations[name][i] : act})
				sess.run(variationssaveop[name][i])
		bar.update(k+1)
	bar.finish()

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
	genPretrainingSteps = 942962#2498201 # Pre-training ended for 
	discPretrainingSteps = 10000
	if last_saved_step is None:
		last_saved_step = 0
		step = 0
	try:			   
		for j in range(20000000):
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
			if step % 5000 == 0:
				summs = sess.run(summaries)
				fw.add_summary(summs,step)
			if step % 5000 == 0:
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
	#			0			1				2		3			4			5
	modes = ["Generate", "FeatureVis", "Train", "Ablate", "Investigate", "histogram"]
	mode = modes[4]

	if mode == modes[0]: #Generate
		generate(16000*3, "D:\\MAESTRO\\maestro-v1.0.0\\2017\\MIDI-Unprocessed_051_PIANO051_MID--AUDIO-split_07-06-17_Piano-e_3-02_wav--2.wav")
	elif mode == modes[1]: # FeatureVis
		feature_max('dilated_stack', 5, 32)
	elif mode == modes[3]: #ABLATE
		ablate(['dilated_stack'], [0,1,2,5, 10, 11, 12, 13, 20, 25, 30, 35, 40, 45]);
	elif mode == modes[4]: #INVESTIGATE
		investigate(['dilated_stack'], [25], "D:\\MAESTRO\\maestro-v1.0.0\\2017\\MIDI-Unprocessed_051_PIANO051_MID--AUDIO-split_07-06-17_Piano-e_3-02_wav--2.wav")
	elif mode == modes[5]: #HISTOGRAMS
		create_histograms(['dilated_stack'], [0,1,2,5])
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
				print("Softies shape")
				print(np.shape(softies))

				#print(np.shape(arg_maxes))
				#arg_maxes = tf.sign(tf.reduce_max(softies, axis=2, keepdims=True)-softies)
				#arg_maxes = (arg_maxes-1)*(-1)
				#print(np.shape(arg_maxes))
				#newest_sample = softies[:,-1,:]
				#Sample from newest_sample

				scaled_prediction = tf.log(softies) / 0.9 #args.temperature # USE
				loged = tf.log(tf.reduce_sum(tf.exp(scaled_prediction), axis=2, keepdims = True)) #USE
				#print(np.shape(loged))
				scaled_prediction = (scaled_prediction - loged) #USE
				#print(np.shape(scaled_prediction))
				#scaled_prediction = tf.exp(scaled_prediction) 
				mask_indexes = tf.multinomial(tf.reshape(scaled_prediction[:,0,:], [g.options["batch_size"], g.options["quantization_channels"]]), 1) #USE
				#print(np.shape(mask_indexes))
				#mask_index = np.random.choice(
				#		np.arange(g.options["quantization_channels"]), p=scaled_prediction)
				
				mask = Generator._one_hot(mask_indexes) #USE
				arg_maxes = tf.sign(softies * mask) #USE

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
				#fake_sample = tf.concat((tf.slice(audio, [0,1,0], [-1,-1,-1]), softies),1)   # if doing input with probability distribution
				fake_sample = tf.concat((tf.slice(audio, [0,1,0], [-1,-1,-1]), arg_maxes),1)    # use
				print("fake sample shape")
				print(np.shape(fake_sample))
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
			e = tf.random_uniform([g.options["batch_size"]],0,1) 
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


