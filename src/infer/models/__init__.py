import importlib

class ModelLoader:
    def __init__(self, model_name, config, use_accel):
        self.model_name = model_name
        self.config = config
        self._model = None
        self.use_accel = use_accel

    def _lazy_import(self, module_name, func_name):
        """Dynamically import a module and return the desired function."""
        if module_name.startswith('.'):
            # Convert relative import to absolute import based on the current package context
            module_name = __package__ + module_name
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    @property
    def model(self):
        """Load and return the model instance, if not already loaded."""
        if self._model is None:
            load_func = self._lazy_import(self.config['load'][0], self.config['load'][1])
            if self.config.get('call_type') == 'api':
                self._model = load_func(
                    self.config['model_path_or_name'], 
                    self.config['base_url'], 
                    self.config['api_key'], 
                    self.config['model']
                )
            else:
                self._model = load_func(self.model_name, self.config, use_accel=self.use_accel)
        return self._model

    @property
    def infer(self):
        """Return the inference function."""
        return self._lazy_import(self.config['infer'][0], self.config['infer'][1])

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, name, config, use_accel):
        """Register a model configuration."""
        self.models[name] = ModelLoader(name, config, use_accel)

    def load_model(self, choice, use_accel=False):
        """Load a model based on the choice."""
        if choice in self.models:
            return self.models[choice].model
        else:
            raise ValueError(f"Model choice '{choice}' is not supported.")

    def infer(self, choice):
        """Get the inference function for a given model."""
        if choice in self.models:
            return self.models[choice].infer
        else:
            raise ValueError(f"Inference choice '{choice}' is not supported.")

# Initialize model registry
model_registry = ModelRegistry()

# Configuration of models
model_configs = {
    'blip2-flan-t5-xl': {
        'load': ('.blip2-flan-t5-xl', 'load_model'),
        'infer': ('.blip2-flan-t5-xl', 'infer'),
        'model_path_or_name': '/path/to/blip2-flan-t5-xl',
        'call_type': 'local',
        'tp': 1
    },
    'qwen-vl-max-latest': {
        'load': ('.api', 'load_model'),
        'infer': ('.api', 'infer'),
        'model_path_or_name': 'qwen-vl-max-latest',
        'base_url': "https://api.example.com/v1/chat/completions",
        'api_key': "sk-your-api-key-here",
        'model': 'qwen-vl-max-latest',
        'call_type': 'api'
    },
    'gpt4o': { 
        'load': ('.api', 'load_model'),
        'infer': ('.api', 'infer'),
        'model_path_or_name': 'gpt-4o',
        'base_url': 'https://api.example.com/v1/chat/completions',
        'api_key': 'sk-your-api-key-here',
        'model': 'gpt-4o',
        'call_type': 'api'
    },
    'claude-3-7-sonnet-20250219': {
        'load': ('.api', 'load_model'),
        'infer': ('.api', 'infer'),
        'model_path_or_name': 'claude-3-7-sonnet-20250219',
        'base_url': 'https://api.example.com/v1/chat/completions',
        'api_key': 'sk-your-api-key-here',
        'model': 'claude-3-7-sonnet-20250219',
        'call_type': 'api'
    },
    'InternVL2-8B': {
        'load': ('.lmdeploy_chat', 'load_model'),
        'infer': ('.lmdeploy_chat', 'infer'),
        'model_path_or_name': '/path/to/models/InternVL2-8B',
        'call_type': 'local',
        'tp': 1
    },
    'InternVL2_5-8B': {
        'load': ('.lmdeploy_chat', 'load_model'),
        'infer': ('.lmdeploy_chat', 'infer'),
        'model_path_or_name': '/path/to/models/OpenGVLab/InternVL2_5-8B',
        'call_type': 'local',
        'tp': 1
    },
    'deepseek_vl2': {
        'load': ('.deepseek_vl2', 'load_model'),
        'infer': ('.deepseek_vl2', 'infer'),
        'model_path_or_name': '/path/to/models/deepseek-ai/deepseek-vl2',
        'call_type': 'local',
        'tp': 1
    },
    'InternLM-XComposer2-VL-7b': {
        'load': ('.InternLM-XComposer2-VL-7b', 'load_model'),
        'infer': ('.InternLM-XComposer2-VL-7b', 'infer'),
        'model_path_or_name': '/path/to/models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b',
        'call_type': 'local',
        'tp': 1
    },
    'Qwen-VL-Chat': {
        'load': ('.qwen_vl_chat', 'load_model'),
        'infer': ('.qwen_vl_chat', 'infer'),
        'model_path_or_name': '/path/to/models/Qwen-VL-Chat',
        'call_type': 'local',
        'tp': 1
    },
    'yi_vl_6b_chat': {
        'load': ('.yi_vl_6b_chat', 'load_model'),
        'infer': ('.yi_vl_6b_chat', 'infer'),
        'model_path_or_name': '/path/to/models/01ai/Yi-VL-6B',
        'call_type': 'local',
        'tp': 1
    },
    'yi_vl_34b_chat': {
        'load': ('.yi_vl_34b_chat', 'load_model'),
        'infer': ('.yi_vl_34b_chat', 'infer'),
        'model_path_or_name': '/path/to/models/01ai/Yi-VL-34B',
        'call_type': 'local',
        'tp': 1
    },
    'idefics2-8b': {
        'load': ('.idefics2', 'load_model'),
        'infer': ('.idefics2', 'infer'),
        'model_path_or_name': '/path/to/models/LLM-Research/idefics2-8b',
        'call_type': 'local',
        'tp': 1
    },
    'InstructBLIP-T5-XL': {
        'load': ('.InstructBLIP-T5-XL', 'load_model'),
        'infer': ('.InstructBLIP-T5-XL', 'infer'),
        'model_path_or_name': '/path/to/models/instructblip-flan-t5-xl',
        'call_type': 'local',
        'tp': 1
    },
    'blip2-flan-t5-xl': {
        'load': ('.blip2-flan-t5-xl', 'load_model'),
        'infer': ('.blip2-flan-t5-xl', 'infer'),
        'model_path_or_name': '/path/to/models/blip2-flan-t5-xl',
        'call_type': 'local',
        'tp': 1
    },
    'visualglm-6b': {
        'load': ('.visualglm-6b', 'load_model'),
        'infer': ('.visualglm-6b', 'infer'),
        'model_path_or_name': '/path/to/models/ZhipuAI/visualglm-6b',
        'call_type': 'local',
        'tp': 1
    },
    'llava_v1_6_34b': {
        'load': ('.llava_v1_6_34b', 'load_model'),
        'infer': ('.llava_v1_6_34b', 'infer'),
        'model_path_or_name': '/path/to/models/llava-hf/llava-v1.6-34b-hf',
        'call_type': 'local',
        'tp': 1
    },
}


def load_model(choice, use_accel=False):
    """Load a specific model based on the choice."""
    model_registry.register_model(choice, model_configs[choice], use_accel)
    return model_registry.load_model(choice, use_accel)

def infer(choice):
    """Get the inference function for a specific model."""
    return model_registry.infer(choice)

