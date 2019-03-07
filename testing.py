import tensorflow as tf
from WaveNET import model as model
import options as o
import loader
import numpy as np

if __name__ == "__main__":
	fw = tf.summary.FileWriter("./tfb_logs/")
	l = loader.AudioReader("maestro-v1.0.0", o.options["sample_rate"], 3)

	#for audio, filename in l.loadAudio("maestro-v1.0.0/2017/"):
	#	print(filename)
	#	print(audio)
	
	Generator = model.WaveNetModel(o.options["batch_size"],
		dilations=o.options["dilations"],
        filter_width=o.options["filter_width"],
        residual_channels=o.options["residual_channels"],
        dilation_channels=o.options["dilation_channels"],
        skip_channels=o.options["skip_channels"],
        quantization_channels=o.options["quantization_channels"],
        use_biases=o.options["use_biases"],
        scalar_input=o.options["scalar_input"],
        initial_filter_width=o.options["initial_filter_width"])

	Discriminator = model.WaveNetModel(o.options["batch_size"],
		dilations=o.options["dilations"],
        filter_width=o.options["filter_width"],
        residual_channels=o.options["residual_channels"],
        dilation_channels=o.options["dilation_channels"],
        skip_channels=o.options["skip_channels"],
        quantization_channels=o.options["quantization_channels"],
        use_biases=o.options["use_biases"],
        scalar_input=o.options["scalar_input"],
        initial_filter_width=o.options["initial_filter_width"])


	#output = Generator._create_network(X, L)
	
	graph = tf.get_default_graph()
	fw.add_graph(graph)

