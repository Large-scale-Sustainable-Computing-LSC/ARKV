from typing import Optional

from transformers import DynamicCache

from akcb.cache.window_cache import WindowCache, WindowDecodingKVCompressorLayerWise, WindowPrefillKVCompressor
from akcb.config import ADCacheConfig

from akcb.cache.mix_cache import MixCache, MixDecodingKVCompressorLayerWise, MixPrefillKVCompressor
from akcb.cache.origin_cache import OriginCache, OriginDecodingKVCompressorLayerWise, OriginPrefillKVCompressor
from akcb.cache.quant_cache import QuantCache, QuantDecodingKVCompressorLayerWise, QuantPrefillKVCompressor



class AdaptiveCache:
    def __init__(self, config: ADCacheConfig):
        self.config = config
        self._init_cache()
        self._init_prefill_compressor()
        self._init_decoding_compressor()

    def cache(self):
        return self.cache
    
    def prefill_cpr(self):
        return self.prefill_compressor
    
    def decoding_cpr(self):
        return self.decoding_compressor

    def _init_cache(self):
        config = self.config
        if config.quant_type == "mix":
            self.cache = MixCache
        elif config.quant_type == "quant":
            self.cache = QuantCache
        elif config.quant_type == "origin":
            self.cache = OriginCache
        elif config.quant_type == "window":
            self.cache = WindowCache
        else:
            raise ValueError(f"Unknown quant_type: {config.quant_type}")

    def _init_prefill_compressor(self):
        config = self.config
        if config.quant_type == "mix":
            self.prefill_compressor = MixPrefillKVCompressor
        elif config.quant_type == "quant":
            self.prefill_compressor = QuantPrefillKVCompressor
        elif config.quant_type == "origin":
            self.prefill_compressor = OriginPrefillKVCompressor
        elif config.quant_type == "window":
            self.prefill_compressor = WindowPrefillKVCompressor
        else:
            raise ValueError(f"Unknown quant_type: {config.quant_type}")

    def _init_decoding_compressor(self):
        config = self.config
        if config.quant_type == "mix":
            self.decoding_compressor = MixDecodingKVCompressorLayerWise
        elif config.quant_type == "quant":
            self.decoding_compressor = QuantDecodingKVCompressorLayerWise
        elif config.quant_type == "origin":
            self.decoding_compressor = OriginDecodingKVCompressorLayerWise
        elif config.quant_type == "window":
            self.decoding_compressor = WindowDecodingKVCompressorLayerWise
        else:
            raise ValueError(f"Unknown quant_type: {config.quant_type}")

    