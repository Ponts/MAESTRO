import tensorflow as tf
from WaveNET import model as model
import options as o
import loader
import numpy as np
from WaveNET import ops 
import progressbar
from time import sleep
import os, sys
import librosa


logdir = "./tfb_logs/"

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
	sr = o.options["sample_rate"]

	with tf.variable_scope("GEN/"):
		Generator = model.WaveNetModel(1,
			dilations=o.options["dilations"],
			filter_width=o.options["filter_width"],
			residual_channels=o.options["residual_channels"],
			dilation_channels=o.options["dilation_channels"],
			skip_channels=o.options["skip_channels"],
			quantization_channels=o.options["quantization_channels"],
			use_biases=o.options["use_biases"],
			scalar_input=o.options["scalar_input"],
			initial_filter_width=o.options["initial_filter_width"],
			global_condition_channels=o.options["noise_dimensions"],
			global_condition_cardinality=None,
			histograms=True)

	sampleph = tf.placeholder(tf.float32, [1,Generator.receptive_field,1])
	noiseph = tf.placeholder(tf.float32, [1,1,o.options["noise_dimensions"]])
	next_sample = Generator._create_network(sampleph, noiseph)
	fake_sample = tf.concat((tf.slice(sampleph, [0,1,0], [-1,-1,-1]), next_sample),1)

	# Get the graph
	variables_to_restore = {
		var.name[:-2]: var for var in tf.global_variables()
		if not ('state_buffer' in var.name or 'pointer' in var.name) and "GEN/" in var.name}
	#print(len(variables_to_restore))

	saver = tf.train.Saver(variables_to_restore)
	print("Restoring model")
	ckpt = tf.train.get_checkpoint_state(logdir)
	#saver.restore(sess, ckpt.model_checkpoint_path)
	saver.restore(sess, "D:\\MAESTRO\\tfb_logs\\model.ckpt-35368")
	print("Model {} restored".format(ckpt.model_checkpoint_path))


	generated = []
	if conditionOn is not None:
		audio, sr = librosa.load(conditionOn, o.options["sample_rate"], mono=True)
		start = np.random.randint(0,len(audio)-Generator.receptive_field)
		fakey = audio[start:start+Generator.receptive_field]
		#fakey = sess.run(audio)
		#generated = fakey.tolist()
	else:
		fakey = [0.0] * (Generator.receptive_field-1)
		fakey.append(np.random.uniform())
	noise = np.random.normal(o.options["noise_mean"], o.options["noise_variance"], size=o.options["noise_dimensions"]).reshape(1,1,-1)
	fakey = np.reshape(fakey, [1,-1,1])
	bar = progressbar.ProgressBar(maxval=length, \
		widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	for i in range(length):
		#print(np.shape(fakey), np.shape(noise))
		#sample = sess.run(next_sample, feed_dict={sampleph : fakey, noiseph : noise})
		fakey = sess.run(fake_sample, feed_dict={sampleph : fakey, noiseph : noise})
		print(fakey[-1,-1,-1])
		#print(np.shape(fakey))
		generated.append(fakey[-1,-1,-1])
		bar.update(i+1)
	bar.finish()
	print(np.shape(generated))
	print(type(generated[0]))
	generated = sess.run(ops.mu_law_decode(generated,256))
	generated = np.array(generated)
	print(np.shape(generated))
	print(type(generated[0]))
	print(generated)
	librosa.output.write_wav("Generated/gangen.wav", generated, sr, norm=True)
	print("Wrote file " + filename + ".")
	




def train(coord, G, D, loader):
	# Get the graph
	conf = tf.ConfigProto(log_device_placement=False)
	conf.gpu_options.allow_growth=True
	sess = tf.Session(config=conf)
	init = tf.global_variables_initializer()
	sess.run(init)
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
	#sample = sess.run(loader.deque(o.options["batch_size"]))
	#print(loader.deque(o.options["batch_size"]))
	#for i in range(recField):
	#	sam = sess.run(fake_sample)
	step = last_global_step
	last_saved_step = last_global_step
	if last_saved_step is None:
		last_saved_step = 0
		step = 0

	try:
		for j in range(1000000):
			if step < 50000: # Do Non-gan init training
				_, lossMl, = sess.run([mlStep, mlLoss])
				if step % 10 == 0:
					_, dLoss = sess.run([discStep, discLoss])
					#base, gen, target = sess.run([mldequed, ml_one_step, mltarget])
					print("Disc Loss : " + str(dLoss))
				print("MLLoss GEN: " + str(lossMl) + ", step: " + str(step))
				step += 1
			else: # Do gan-training
				loader.mlDone = True

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
				fakey = sess.run(audio)
				init_noise = sess.run(noise)
				#for k in range(recField):
				fakey = sess.run(fake_sample, feed_dict={audio : fakey, noise : init_noise})
				_, dLoss, gLoss, slopeLoss, gradsLoss = sess.run([genStep, discLoss, genLoss, nr, slope], feed_dict={fake_sample : fakey, noise : init_noise})
				print("DiscLoss:  " + str(dLoss))
				#print("SlopeLoss: " + str(slopeLoss))
				#print("GradsLoss: " + str(gradsLoss))
				print("GenLoss:   " + str(gLoss) + ", Step: " + str(step))
			#	print()
				step += 1
			if step % 500 == 0:
				print(step)
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
	generate(5147, "D:\\MAESTRO\\maestro-v1.0.0\\2017\\MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--1.wav")
	
	if True:
		print("Done")
	else:


		fw = tf.summary.FileWriter(logdir)
		coord = tf.train.Coordinator()

		with tf.variable_scope("GEN/"):
			Generator = model.WaveNetModel(o.options["batch_size"],
				dilations=o.options["dilations"],
				filter_width=o.options["filter_width"],
				residual_channels=o.options["residual_channels"],
				dilation_channels=o.options["dilation_channels"],
				skip_channels=o.options["skip_channels"],
				quantization_channels=o.options["quantization_channels"],
				use_biases=o.options["use_biases"],
				scalar_input=o.options["scalar_input"],
				initial_filter_width=o.options["initial_filter_width"],
				global_condition_channels=o.options["noise_dimensions"],
				global_condition_cardinality=None,
				histograms=True)


		with tf.variable_scope("DIS/"):
			Discriminator = model.WaveNetModel(o.options["batch_size"],
				dilations=o.options["dilations"],
				filter_width=o.options["filter_width"],
				residual_channels=o.options["residual_channels"],
				dilation_channels=o.options["dilation_channels"],
				skip_channels=o.options["skip_channels"],
				quantization_channels=o.options["quantization_channels"],
				use_biases=o.options["use_biases"],
				scalar_input=o.options["scalar_input"],
				initial_filter_width=o.options["initial_filter_width"],
				global_condition_channels=None,
				global_condition_cardinality=None,
				histograms = True)
		# Audio dimensions: [Song, Samplenr, Quantization]
		# Data loading
		with tf.name_scope("Pre-processing"):
			l = loader.AudioReader("maestro-v1.0.0/2017", o.options["sample_rate"], Discriminator.receptive_field, coord, stepSize=1, sampleSize=o.options["sample_size"])
			abatch = l.deque(o.options["batch_size"])
			mldequed = l.dequeMl(o.options["batch_size"])
			mldequed = tf.reshape(tf.cast(mldequed, tf.float32), [o.options["batch_size"], -1, 1])
			mltarget = tf.slice(mldequed,[0,Generator.receptive_field,0],[-1,-1,-1])
			mlinput = tf.slice(mldequed,[0,0,0],[-1, tf.shape(mldequed)[1]-1, -1])
			encoded = tf.reshape(tf.cast(abatch, tf.float32), [o.options["batch_size"], -1, 1])
			codedNoise = l.dequeNoise(o.options["batch_size"])

		# Generator stuff
		with tf.variable_scope("GEN/", reuse=tf.AUTO_REUSE):
			audio = encoded
			noise = Generator._embed_gc(codedNoise)
			generated = Generator._create_network(audio, noise)
			mlGen = Generator._create_network(mlinput, noise)
			with tf.name_scope("Generating/"):
				ml_one_step = tf.reshape(mlGen, [o.options["batch_size"],-1,1])
				one_step = tf.reshape(generated,[o.options["batch_size"],-1,1]) 
				# Shift the generated vector
				fake_sample = tf.concat((tf.slice(audio, [0,1,0], [-1,-1,-1]), one_step),1)

		
		# Discriminator stuff
		with tf.variable_scope("DIS/", reuse=tf.AUTO_REUSE):
			daudio = encoded
			r_logits = Discriminator._create_network(daudio, None)
			#r_logits = tf.layers.dense(real_output, 1, name="final_D")
		with tf.variable_scope("DIS/", reuse=True):
			f_logits = Discriminator._create_network(fake_sample, None)
			#f_logits = tf.layers.dense(fake_output, 1, name="final_D")

		# Get the variables
		genVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GEN/")
		discVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DIS/")

		with tf.name_scope("GP/"):
			# GP
			e = tf.random_uniform([o.options["batch_size"]],0,1) #WRONG
			e = tf.reshape(e, (o.options["batch_size"],-1,1))
			e = tf.tile(e, [1,tf.shape(daudio)[1],1])
			polated = tf.add(tf.multiply(daudio,e), tf.multiply(fake_sample,1-e))

		with tf.variable_scope("DIS/", reuse=True):
			discPolated = Discriminator._create_network(polated, None)
			
		with tf.name_scope("GP/"):
			grads = tf.gradients(discPolated,polated)[0]
			slope = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1,2]))
			nr = 10*tf.reduce_mean((slope-1)**2)


		with tf.name_scope("Loss/"):
			discLoss = -tf.reduce_mean(r_logits) + tf.reduce_mean(f_logits) #+ 10*tf.reduce_mean((slope-1)**2)
			genLoss = -tf.reduce_mean(f_logits)
			mlLoss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=ml_one_step, labels=mltarget))



		mlStep = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0, beta2=0.9).minimize(mlLoss, var_list=genVars)
		genStep = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0, beta2=0.9).minimize(genLoss, var_list=genVars)
		discStep = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0, beta2=0.9).minimize(discLoss, var_list=discVars)
		
		graph = tf.get_default_graph()
		fw.add_graph(graph)

		train(coord, Generator, Discriminator, l)

		#generate(generated, noise, 16000, Discriminator.receptive_field, audio, l)

