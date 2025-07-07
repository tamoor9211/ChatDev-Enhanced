# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

import os
from camel.typing import ModelType

# Centralized model configuration using environment variables
class ModelConfig:
    """Centralized model configuration management using environment variables."""
    
    # Default model types - can be overridden by environment variables
    DEFAULT_MODEL = os.getenv('CHATDEV_DEFAULT_MODEL', 'GEMINI_2_0_FLASH_EXP')
    FAST_MODEL = os.getenv('CHATDEV_FAST_MODEL', 'GEMINI_1_5_FLASH')
    POWERFUL_MODEL = os.getenv('CHATDEV_POWERFUL_MODEL', 'GEMINI_1_5_PRO')
    VISION_MODEL = os.getenv('CHATDEV_VISION_MODEL', 'GEMINI_PRO_VISION')
    
    # Model type mappings
    _MODEL_MAPPING = {
        'GEMINI_1_5_FLASH': ModelType.GEMINI_1_5_FLASH,
        'GEMINI_1_5_PRO': ModelType.GEMINI_1_5_PRO,
        'GEMINI_1_0_PRO': ModelType.GEMINI_1_0_PRO,
        'GEMINI_PRO': ModelType.GEMINI_PRO,
        'GEMINI_PRO_VISION': ModelType.GEMINI_PRO_VISION,
        'GEMINI_2_0_FLASH_EXP': ModelType.GEMINI_2_0_FLASH_EXP,
        # Backward compatibility
        'GPT_3_5_TURBO': ModelType.GEMINI_1_5_FLASH,
        'GPT_4': ModelType.GEMINI_1_5_PRO,
        'GPT_4_TURBO': ModelType.GEMINI_1_5_PRO,
        'GPT_4O': ModelType.GEMINI_1_5_PRO,
        'GPT_4O_MINI': ModelType.GEMINI_1_5_FLASH,
    }
    
    @classmethod
    def get_default_model(cls) -> ModelType:
        """Get the default model type from environment or fallback."""
        return cls._MODEL_MAPPING.get(cls.DEFAULT_MODEL, ModelType.GEMINI_2_0_FLASH_EXP)
    
    @classmethod
    def get_fast_model(cls) -> ModelType:
        """Get the fast model type from environment or fallback."""
        return cls._MODEL_MAPPING.get(cls.FAST_MODEL, ModelType.GEMINI_1_5_FLASH)
    
    @classmethod
    def get_powerful_model(cls) -> ModelType:
        """Get the powerful model type from environment or fallback."""
        return cls._MODEL_MAPPING.get(cls.POWERFUL_MODEL, ModelType.GEMINI_1_5_PRO)
    
    @classmethod
    def get_vision_model(cls) -> ModelType:
        """Get the vision model type from environment or fallback."""
        return cls._MODEL_MAPPING.get(cls.VISION_MODEL, ModelType.GEMINI_PRO_VISION)
    
    @classmethod
    def get_model_by_name(cls, model_name: str) -> ModelType:
        """Get model type by name with fallback to default."""
        return cls._MODEL_MAPPING.get(model_name, cls.get_default_model())
    
    @classmethod
    def set_environment_defaults(cls):
        """Set environment variables for consistent model usage."""
        os.environ.setdefault('CHATDEV_DEFAULT_MODEL', cls.DEFAULT_MODEL)
        os.environ.setdefault('CHATDEV_FAST_MODEL', cls.FAST_MODEL)
        os.environ.setdefault('CHATDEV_POWERFUL_MODEL', cls.POWERFUL_MODEL)
        os.environ.setdefault('CHATDEV_VISION_MODEL', cls.VISION_MODEL)

# Initialize environment defaults
ModelConfig.set_environment_defaults()

# Convenience functions for backward compatibility
def get_default_model_type() -> ModelType:
    """Get the default model type - replaces ModelType.GPT_3_5_TURBO references."""
    return ModelConfig.get_default_model()

def get_fast_model_type() -> ModelType:
    """Get the fast model type."""
    return ModelConfig.get_fast_model()

def get_powerful_model_type() -> ModelType:
    """Get the powerful model type."""
    return ModelConfig.get_powerful_model()

def get_vision_model_type() -> ModelType:
    """Get the vision model type."""
    return ModelConfig.get_vision_model()
