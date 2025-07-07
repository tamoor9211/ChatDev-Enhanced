# Migration from OpenAI to Google AI Studio

This document describes the migration from OpenAI API to Google AI Studio API (Gemini models) in ChatDev.

## Changes Made

### 1. Dependencies
- **Removed**: `openai==1.47.1`
- **Added**: `google-generativeai==0.8.3`
- **Added**: `sentence-transformers==2.2.2` (for local embedding fallback)
- **Removed**: `tiktoken==0.8.0` (not needed for Gemini)

### 2. Model Types
The following model mappings have been implemented:

| Old OpenAI Model | New Gemini Model |
|------------------|------------------|
| GPT-3.5-turbo    | gemini-1.5-flash |
| GPT-4            | gemini-1.5-pro   |
| GPT-4-turbo      | gemini-1.5-pro   |
| GPT-4o           | gemini-1.5-pro   |
| GPT-4o-mini      | gemini-1.5-flash |

### 3. Environment Variables
- **Old**: `OPENAI_API_KEY`
- **New**: `GOOGLE_API_KEY` (with backward compatibility for `OPENAI_API_KEY`)

### 4. Key Files Modified

#### Core Model Backend
- `camel/model_backend.py`: Replaced `OpenAIModel` with `GoogleAIModel`
- `camel/typing.py`: Updated `ModelType` enum with Gemini models
- `camel/utils.py`: Updated token counting and API key validation

#### Embedding System
- `ecl/embedding.py`: Replaced OpenAI embeddings with hybrid Google AI + local fallback
- `ecl/local_embedding.py`: **NEW** - Local embedding system using sentence-transformers
- `ecl/config.yaml`: Changed embedding method to "GoogleAI"
- `ecl/utils.py`: Updated utility functions for Google AI

#### Local Embedding Fallback
- **Added** `ecl/local_embedding.py` with three embedding classes:
  - `LocalEmbedding`: Uses sentence-transformers (all-MiniLM-L6-v2, ~90MB)
  - `HybridEmbedding`: Tries Google AI first, falls back to local model
  - Automatic fallback hierarchy: Google AI → Local Model → Random (last resort)
- **Lightweight**: all-MiniLM-L6-v2 model works well on 8GB RAM
- **Offline capable**: Works without internet connection when Google AI fails

#### Agent System
- `camel/agents/chat_agent.py`: Updated to use Google API key decorator
- `camel/messages/base.py`: Removed OpenAI-specific imports

#### Other Components
- `run.py`: Updated model type mappings
- `chatdev/statistics.py`: Updated cost calculations for Gemini pricing
- `camel/web_spider.py`: Updated to use Google AI for text generation
- `chatdev/eval_quality.py`: Updated embedding functions
- `chatdev/chat_env.py`: Disabled image generation (not available in Google AI Studio)

### 5. Token Counting
Since Google AI Studio doesn't use tiktoken, we implemented a word-based approximation:
- 1 token ≈ 0.75 words (multiplier of 1.3)
- This provides reasonable estimates for cost calculation and token limits

### 6. Embedding Fallback System
The new embedding system provides robust fallback options:
1. **Primary**: Google AI Studio embeddings (models/text-embedding-004)
2. **Fallback**: Local sentence-transformers model (all-MiniLM-L6-v2)
3. **Last Resort**: Random embeddings (for development/testing)

**Benefits**:
- Works offline when Google AI is unavailable
- Lightweight local model (~90MB) suitable for 8GB RAM
- Maintains embedding quality with open-source alternatives
- Automatic failover without user intervention

### 6. Cost Calculation
Updated pricing based on Google AI Studio rates:
- Gemini 1.5 Flash: $0.075/$0.30 per 1M tokens (input/output)
- Gemini 1.5 Pro: $1.25/$5.00 per 1M tokens (input/output)
- Gemini 1.0 Pro: $0.50/$1.50 per 1M tokens (input/output)

### 7. Image Generation
Image generation has been disabled as Google AI Studio doesn't provide image generation APIs. The system will log requests but skip actual generation.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
```bash
export GOOGLE_API_KEY="your_google_ai_studio_api_key"
```

### 3. Run ChatDev
```bash
python run.py --task "your task" --name "project_name" --model GEMINI_1_5_FLASH
```

## Available Models
- `GEMINI_1_5_FLASH`: Fast, cost-effective model
- `GEMINI_1_5_PRO`: High-performance model
- `GEMINI_1_0_PRO`: Standard model
- `GEMINI_PRO`: Alias for GEMINI_1_0_PRO
- `GEMINI_PRO_VISION`: Vision-capable model

## Backward Compatibility
The system maintains backward compatibility:
- Old model names (GPT_4, GPT_3_5_TURBO) are mapped to appropriate Gemini models
- `OPENAI_API_KEY` environment variable is still accepted as `GOOGLE_API_KEY`
- All existing configuration files and scripts should work without modification

## Notes
- Token counting is approximate and may differ from actual usage
- Image generation is not supported
- Function calling and tool usage features are simplified
- Cost calculations are estimates based on Google AI Studio pricing

## Getting Google AI Studio API Key
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Create a new API key
4. Set the `GOOGLE_API_KEY` environment variable
