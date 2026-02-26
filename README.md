# XAVI AI Story Pipeline 🚀

A production-grade pipeline for generating **personalized children's picture books** using Google Gemini 2.5 Flash. Outputs fully formatted Word documents with AI-generated illustrations and age-appropriate text.

---

## ✨ Key Features

- **📖 Full XAVI System Prompt** — Child-safe storytelling with safety protocol, silent reframing, age-based parameters, and visual consistency enforcement
- **🎯 Age-Adaptive Output** — 8 pages (ages 3-5), 12 pages (ages 6-8), 18 pages (ages 9-12) with strict word count limits
- **🎨 AI Image Generation** — Per-page illustrations via Imagen with character consistency lock and auto-repair guardrails
- **📄 Word Document Output** — Side-by-side layout (image left, text right) with cover page
- **⚡ Response Caching** — Same prompt never hits the API twice
- **📉 Token Budgeting** — Soft-limit guardrails that auto-downgrade when 80% budget is spent
- **📊 Detailed Analytics** — Per-call tracking of tokens, latency, and estimated USD cost based on latest paid-tier pricing

---

## 🛠️ Setup

### 1. Prerequisites
- Python 3.9+
- A Google Gemini API Key ([aistudio.google.com](https://aistudio.google.com/))

### 2. Installation
```powershell
cd story-ai-pipeline

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Config
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your_api_key_here"
IMAGE_DELAY_SECONDS=4
```

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `generate.py` | **Main CLI entry point** — generates stories with Word docs |
| `story_generator.py` | XAVI system prompt + user prompt builder → Gemini Flash |
| `doc_builder.py` | Image generation + Word document assembly |
| `gemini_client.py` | Async Gemini SDK client with retry, cache, rate limiting |
| `router.py` | Model routing (Flash / Pro / Imagen classification) |
| `rate_limiter.py` | Token budget management per session |
| `cache.py` | Persistent file-based response caching |
| `logger.py` | Structured JSONL logging for every API call |
| `config.py` | Central configuration (models, limits, thresholds) |
| `image_generator.py` | Imagen integration for batch_runner |
| `batch_runner.py` | Async batch processor (alternative to generate.py) |
| `analytics.py` | Post-run cost and token analytics |
| `diagnostics.py` | Internal stress tests (rate limiter, concurrency) |
| `smoke_test.py` | Module verification (6 modules) |
| `PIPELINE.md` | Detailed technical documentation |

---

## 📊 Story Configuration

| Dimension | Count | Examples |
|-----------|-------|---------|
| **Child Profiles** | 10 | Leo (3), Rose (4), Noah (6), Olivia (11), Alex (12) |
| **Categories** | 13 | Adventure, Fantasy, Friendship, Mythology, Sci-Fi |
| **Tones** | 8 | Happy, Calm, Exciting, Funny, Inspirational |
| **Moral Values** | 8 | Honesty, Kindness, Sharing, Forgiveness, Patience |
| **Art Styles** | 5 | Disney3D, Watercolor, FlatVector, Anime, Clay |
| **Prompts** | 95 | "bedtime story about dinosaur", "story about brave girl" |
| **Total Combos** | **4,742,400** | Unique story configurations |

---

## 🧠 Models

| Purpose | Model | Status |
|---------|-------|--------|
| Story text generation | `gemini-2.5-flash` | ✅ Active (Paid Tier) |
| Image generation | `imagen-4.0-fast` | ✅ Active |
| Complex/long tasks | `gemini-1.5-pro` | ⏸️ Disabled (cost savings) |

---

## 💰 Cost Estimates

| Batch Size | Text Cost (Paid Tier) | Time (text only) | Time (with images) |
|------------|-----------|-------------------|---------------------|
| 1 story | ~$0.007 | ~12s | ~2-5 min |
| 10 stories | ~$0.07 | ~3 min | ~20-50 min |
| 100 stories | ~$0.70 | ~30 min | ~3-8 hours |

---

## 📋 Commands Reference

### Story Generation

| Command | What It Does |
|---------|-------------|
| `python generate.py 1` | Generate 1 story with images + Word doc |
| `python generate.py 5` | Generate 5 stories with full output |
| `python generate.py 100` | Batch generate 100 stories |
| `python generate.py 5 --text-only` | Generate 5 stories (JSON only, no images/docs — fast) |
| `python generate.py 1 --dry-run` | Test run with mock data (no API calls, no cost) |

### Monitoring & Diagnostics

| Command | What It Does |
|---------|-------------|
| `python analytics.py` | Show cost breakdown by model and task type from logs |
| `python diagnostics.py` | Stress test rate limiter, concurrency cap, history compression |
| `python smoke_test.py` | Verify all 6 pipeline modules are working correctly |

### Batch Runner (Alternative)

| Command | What It Does |
|---------|-------------|
| `python batch_runner.py --config configs.json` | Run batch from a pre-built config JSON file |
| `python batch_runner.py --config configs.json --dry-run` | Dry-run a batch (no API calls) |
| `python batch_runner.py --config configs.json --force-fallback` | Test Pro → Flash fallback logic |

---

## 📜 License
MIT
