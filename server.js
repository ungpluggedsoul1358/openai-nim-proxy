// server.js - Optimized OpenAI to NVIDIA NIM API Proxy (without dotenv)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.error("FATAL ERROR: NIM_API_KEY environment variable is not set.");
  console.error("Please set it before running the server.");
  console.error("Example: NIM_API_KEY=your_api_key_here node server.js");
  process.exit(1);
}

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gpt-4': 'meta/llama-3.1-70b-instruct',
  'gpt-4-turbo': 'meta/llama-3.1-70b-instruct',
  'gpt-4o': 'deepseek-ai/deepseek-v2-chat',
  'claude-3-opus': 'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet': 'meta/llama-3.1-70b-instruct',
  'gemini-pro': 'google/gemini-pro'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to NVIDIA NIM Proxy' });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Fallback model logic
    const nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-8b-instruct';

    // FIX: Ensure max_tokens is never too low
    const effective_max_tokens = (max_tokens && max_tokens > 256) ? max_tokens : 4096;

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.7,
      max_tokens: effective_max_tokens,
      stream: stream || false
    };
    
    const nimHeaders = {
      'Authorization': `Bearer ${NIM_API_KEY}`,
      'Content-Type': 'application/json'
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: nimHeaders,
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      response.data.pipe(res);

    } else {
      res.json({
        id: response.data.id || `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: response.data.created || Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => ({
          index: choice.index,
          message: {
            role: choice.message.role,
            content: choice.message.content
          },
          finish_reason: choice.finish_reason
        })),
        usage: response.data.usage
      });
    }
    
  } catch (error) {
    console.error('Error proxying request to NVIDIA NIM API:');
    if (error.response) {
      console.error(`Status: ${error.response.status}`);
      console.error('Data:', JSON.stringify(error.response.data, null, 2));
    } else {
      console.error(error.message);
    }
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.response?.data?.error?.message || error.message || 'An internal server error occurred.',
        type: 'proxy_error',
        code: error.response?.status
      }
    });
  }
});

app.listen(PORT, () => {
  console.log(`✅ OpenAI to NVIDIA NIM Proxy running on http://localhost:${PORT}`);
});
