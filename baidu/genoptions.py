options = {
		"filter_width": 2,
		"sample_rate": 16000,
		"dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
						1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
						1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
						1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
		"residual_channels": 128,
		"dilation_channels": 32,
		"quantization_channels": 256,
		#"skip_channels": 512,
		"skip_channels":256,
		"use_biases": True, #Maybe change?
		"scalar_input": False,
		"initial_filter_width": 32,
		"batch_size" : 8, #32 gives OOM
		"noise_dimensions" : 100,
		"noise_variance" : 0.1,
		"noise_mean" : 0.0,
		"sample_size" : 3000,
		"final_layer_size" : 1,
	}