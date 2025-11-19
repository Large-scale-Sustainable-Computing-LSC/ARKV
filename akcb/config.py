class ADCacheConfig:
    def __init__(self, args):
        self.cache_size = args.cache_size
        self.window_size = args.window_size
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.tau3 = args.tau3
        self.gamma = args.gamma
        self.quant_type = args.quant_type
        self.key_size = args.cache_size - args.window_size
        self.compress = args.compress
        self.print_oq_ratio = True
        
    def update_layer_num(self, layer_num):
        self.layer_num = layer_num
        # self.prefill_compressor = [None] * self.layer_num
        self.refresh_model_settings()

    def refresh_model_settings(self):
        self.decoding_compressor = [None] * self.layer_num
        self.prefill = [True] * self.layer_num
        self.quant_ratio_per_layer = [0.0] * self.layer_num