import tensorflow as tf
from WaveNET import model as model
import options as o
import loader
import numpy as np
from WaveNET import ops 
import progressbar
from time import sleep


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

def train(genStep, discStep, genLoss, discLoss, fake_sample, audio, noise, coord, G, D, loader, one_step, generated, abatch, graph=tf.get_default_graph()):
	# Get the graph
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
	init = tf.global_variables_initializer()
	sess.run(init)
	# Saver
	saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=5)
	try:
		loaded = load(saver, sess, logdir)
		if loaded is not None:
			print("Restored model")
	except:
		print("Error in restoring model, terminating")
		raise
	
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	loader.startThreads(sess)

	
	# Generate samples of the Discriminators receptive field
	samples = []
	recField = D.receptive_field
	print("running fake sample")
	#sample = sess.run(loader.deque(o.options["batch_size"]))
	#print(loader.deque(o.options["batch_size"]))
	#for i in range(recField):
	#	sam = sess.run(fake_sample)
	init_audio, init_noise  = sess.run([audio,noise])


	fakey = sess.run(fake_sample, feed_dict={audio:init_audio, codedNoise:init_noise})
	print(np.shape(fakey))
	print(np.shape(init_audio))
	fakey2 = sess.run(fake_sample, feed_dict={audio:fakey, codedNoise:init_noise})
	

if __name__ == "__main__":
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
		l = loader.AudioReader("maestro-v1.0.0", o.options["sample_rate"], Discriminator.receptive_field, coord, sampleSize=o.options["sample_size"])
		abatch = l.deque(o.options["batch_size"])
		encoded = tf.reshape(tf.cast(abatch, tf.float32), [o.options["batch_size"], -1, 1])
		codedNoise = l.dequeNoise(o.options["batch_size"])

	# Generator stuff
	with tf.variable_scope("GEN/", reuse=tf.AUTO_REUSE):
		audio = encoded
		noise = Generator._embed_gc(codedNoise)
		generated = Generator._create_network(audio, noise)

		with tf.name_scope("Generating/"):
			#bla = tf.multinomial(tf.log(generated), 1)
			#print(np.shape(bla))
			#for i in range(Discriminator.receptive_field):
			# generate one sample			
			one_step = tf.reshape(generated,[o.options["batch_size"],-1,1]) # Gradient disappears
			# Shift the generated vector
			fake_sample = tf.concat((tf.slice(audio, [0,1,0], [-1,-1,-1]), one_step),1)

			#print(i / Discriminator.receptive_field)
			# Draw for each in batch according to temperature
			'''
			#np.seterr(divide='ignore') 			#temp
			scaled_prediction = tf.log(generated) / 0.9
			scaled_prediction = (scaled_prediction -
								tf.tile(tf.reshape(tf.reduce_logsumexp(scaled_prediction, 2),[1,-1,o.options["quantization_channels"]]), [o.options["batch_size"],1,1]))
			scaled_prediction = tf.reshape(tf.exp(scaled_prediction), [o.options["batch_size"],o.options["quantization_channels"]])
			#np.seterr(divide='warn')

			#sample = np.random.choice(np.arange(o.options["quantization_channels"]), o.options["batch_size"], p=scaled_prediction)
			generated = ops.mu_law_decode(tf.reshape(tf.multinomial(tf.log(scaled_prediction),1),[o.options["batch_size"],-1,1]), o.options["quantization_channels"])
			'''
			
	
	# Discriminator stuff
	with tf.variable_scope("DIS/", reuse=tf.AUTO_REUSE):
		daudio = encoded
		real_output = Discriminator._create_network(daudio, None)
		r_logits = tf.layers.dense(real_output, 1, name="final_D")
	with tf.variable_scope("DIS/", reuse=True):
		fake_output = Discriminator._create_network(fake_sample, None)
		f_logits = tf.layers.dense(fake_output, 1, name="final_D")

	# Get the variables
	genVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GEN/")
	discVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DIS/")

	with tf.name_scope("GP/"):
		# GP
		e = tf.random_uniform(tf.shape(daudio),0,1)
		polated = tf.add(tf.multiply(daudio,e), tf.multiply(fake_sample,1-e))
	with tf.variable_scope("DIS/", reuse=True):
		interm = Discriminator._create_network(polated, None)
		discPolated = tf.layers.dense(interm, 1, name="final_D")
	with tf.name_scope("GP/"):
		grads = tf.gradients(discPolated,[polated])
		#print((grads))
		slope = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))

	with tf.name_scope("Loss/"):
		discLoss = -tf.reduce_mean(r_logits) + tf.reduce_mean(f_logits) + 10*tf.reduce_mean((slope-1)**2)
		genLoss = -tf.reduce_mean(f_logits)


	genStep = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0, beta2=0.9).minimize(genLoss, var_list=genVars)
	discStep = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0, beta2=0.9).minimize(discLoss, var_list=discVars)
	
	graph = tf.get_default_graph()
	fw.add_graph(graph)

	print(audio)
	train(genStep, discStep, genLoss, discLoss, fake_sample, audio, noise, coord, Generator, Discriminator, l, one_step, generated, abatch, graph)



