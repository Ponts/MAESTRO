options = {
		"filter_width": 2,
		"sample_rate": 16000,
		"dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
						1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
						1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
						1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
						1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
		"residual_channels": 32,
		"dilation_channels": 32,
		"quantization_channels": 256,
		"skip_channels": 512,
		"use_biases": True,
		"scalar_input": False,
		"initial_filter_width": 32,
		"batch_size" : 128
	}