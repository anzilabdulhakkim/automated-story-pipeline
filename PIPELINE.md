# Prajna Story Pipeline — Technical Documentation

## Overview

This pipeline generates **personalized, age-appropriate children's picture books** as Word documents. Each story includes AI-generated text and illustrations, formatted in a side-by-side (image + text) layout.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         generate.py                             │
│  CLI entry point — builds random configs, orchestrates pipeline │
└──────────────┬──────────────────────────────────┬───────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────┐        ┌──────────────────────────┐
│   story_generator.py     │        │     doc_builder.py       │
│  Prajna system prompt    │        │  Image gen + Word doc    │
│  + user prompt builder   │        │  assembly                │
└───────────┬──────────────┘        └───────────┬──────────────┘
            │                                   │
            ▼                                   │
┌──────────────────────────┐                    │
│    gemini_client.py      │                    │
│  Async Gemini SDK client │                    │
│  ┌────────────────────┐  │                    │
│  │ router.py          │  │     Model: gemini-2.5-flash-image
│  │ rate_limiter.py    │  │                    │
│  │ cache.py           │  │                    │
│  │ logger.py          │  │                    │
│  └────────────────────┘  │                    │
│                          │                    │
│  Model: gemini-2.5-flash │                    │
└──────────────────────────┘                    │
                                                │
                    ┌───────────────────────────┘
                    ▼
        ┌───────────────────┐
        │   Output Files    │
        │  .docx + images   │
        │  + analytics JSON │
        └───────────────────┘
```

---

## Execution Flow (Step by Step)

### Step 1 — Config Generation (`generate.py`)

When you run `python generate.py 5`, the pipeline:

1. Generates **5 random story configs** by mixing:
   - A random **child profile** (name, gender, age)
   - A random **user prompt** (story idea)
   - A random **category**, **tone**, **moral**, and **art style**
2. Saves configs to `story_configs_batch_5_YYYYMMDD_HHMMSS.json`
3. Processes each story **sequentially**

### Step 2 — Story Text Generation (`story_generator.py` → `gemini_client.py`)

For each config, the pipeline calls `generate_story()`:

```
1. Build Prajna system prompt (~4,000 tokens)
   └─ Safety rules, age parameters, image prompt structure, JSON schema

2. Build user prompt (~100 tokens)
   └─ "Story Idea: X, Child's Name: Y, Age: Z, Category: A, Tone: B..."

3. Send to gemini_client.generate()
   ├─ rate_limiter.is_exhausted(session_id)?    → Block if budget spent
   ├─ router.route(prompt, task_type)           → Always returns FLASH
   ├─ cache.get(model, system_prompt, prompt)   → Return cached if exists
   ├─ _raw_call() to Gemini API
   │   ├─ Model: gemini-2.5-flash
   │   ├─ response_mime_type: "application/json"
   │   ├─ thinking_config: disabled (budget=0)
   │   └─ max_output_tokens: 8,192
   ├─ cache.set() → Store response for future hits
   ├─ rate_limiter.record_usage() → Track token spend
   └─ logger.log() → Write to logs/api_calls.jsonl
```

**Output**: Full story JSON with `pages[]`, `cover_image_prompt`, `character_description`, etc.

### Step 3 — Document Assembly (`doc_builder.py`)

For each completed story JSON:

```
1. Download cover image
   └─ Model: gemini-2.5-flash-image (1:1 aspect ratio)

2. For each page (8/12/18 depending on age):
   ├─ Download page image from image_prompt
   ├─ 4-second delay between calls (rate limit guard)
   └─ 3 retries with exponential backoff (15s → 30s → 60s)

3. Assemble Word document (python-docx)
   ├─ Title + metadata header
   ├─ Cover image (centered, 5" wide)
   └─ Per page: 2-column table [Image 3.2" | Text]

4. Save to documents/
   └─ Prajna_{age}yr_{name}_{title}.docx
```

---

## Configuration Options

### Child Profiles (10)

| Name | Gender | Age | Age Group |
|------|--------|-----|-----------|
| Leo | he | 3 | 3-5 (8 pages) |
| Rose | she | 4 | 3-5 (8 pages) |
| Mia | she | 5 | 3-5 (8 pages) |
| Noah | he | 6 | 6-8 (12 pages) |
| Emma | she | 7 | 6-8 (12 pages) |
| Sam | they | 8 | 6-8 (12 pages) |
| Liam | he | 9 | 9-12 (18 pages) |
| Ethan | he | 10 | 9-12 (18 pages) |
| Olivia | she | 11 | 9-12 (18 pages) |
| Alex | they | 12 | 9-12 (18 pages) |

### Story Categories (13)

| # | Category |
|---|----------|
| 1 | Family |
| 2 | Friendship |
| 3 | Love |
| 4 | Adventure |
| 5 | Animals |
| 6 | Fantasy |
| 7 | Courage |
| 8 | Imagination |
| 9 | Fairy Tales |
| 10 | Perseverance |
| 11 | Folktales |
| 12 | Mythology |
| 13 | Science Fiction |

### Story Tones (8)

| # | Tone | Effect on Language | Image Color Palette |
|---|------|-------------------|-------------------|
| 1 | Happy | Bright words, upbeat rhythm | Bright primary colors |
| 2 | Sad | Gentle, reflective words | Soft muted tones |
| 3 | Exciting | Power words, quick energy | Intense vivid colors |
| 4 | Calm | Soft words, slow flow | Soft pastels |
| 5 | Funny | Playful exaggeration, twists | Bright mixed colors |
| 6 | Scary | Reframed to "mysterious" (safety) | Earth tones |
| 7 | Educational | Informative, curious | Warm earth tones |
| 8 | Inspirational | Empowering, hopeful | Bold contrasting colors |

### Moral Values (8)

| # | Moral | How It's Taught |
|---|-------|----------------|
| 1 | Honesty | Character chooses truth despite difficulty |
| 2 | Kindness | Character helps others, sees positive result |
| 3 | Sharing | Character shares resources, gains friendship |
| 4 | Forgiveness | Character forgives, relationship improves |
| 5 | Patience | Character waits, achieves better outcome |
| 6 | Responsibility | Character takes ownership, earns trust |
| 7 | Respect | Character treats others well, builds community |
| 8 | Gratitude | Character appreciates what they have |

### Art Styles (6)

| # | Style | Visual Characteristics |
|---|-------|----------------------|
| 1 | Cartoon | Bold, colorful, expressive |
| 2 | Watercolor | Soft brushstrokes, dreamy quality |
| 3 | Sketch | Hand-drawn, pencil texture |
| 4 | Pixel Art | Retro, geometric blocks |
| 5 | Anime | Large eyes, dynamic, manga-inspired |
| 6 | Oil Painting | Rich textures, classic feel |

### User Prompts (95 unique prompts)

Examples: "bedtime story about dinosaur", "story about brave girl", "moral story about honesty", "story about space rocket", "funny monster story", etc.

### Total Unique Combinations

```
10 profiles × 13 categories × 8 tones × 8 morals × 95 prompts × 6 art styles
= 4,742,400 unique story configurations
```

---

## Age-Based Story Parameters

| Parameter | Ages 3-5 | Ages 6-8 | Ages 9-12 |
|-----------|----------|----------|-----------|
| **Total Pages** | 8 | 12 | 18 |
| **Words Per Page** | 25-50 | 40-100 | 75-120 |
| **Total Words** | 250-500 | 560-1,400 | 1,500-2,400 |
| **Vocabulary** | 0-1 new words | 2-3 new words | 3-5 new words |
| **Conflict Level** | Very mild | Moderate | Substantive |
| **Narrative** | Setup → Problem → Success | Setup → Climax → Resolution | Exposition → Climax → Resolution |
| **Camera Angle** | Full body shot | Medium shot | Dynamic medium shot |

---

## Output Files

| File | Location | Persists? |
|------|----------|-----------|
| Word document | `./documents/Prajna_...docx` | ✅ Unique per story |
| Markdown story | `./stories/Prajna_STORY_...md` | ✅ Unique per story |
| Story images | `./story_images/{name}_{title}/page_N.jpg` | ✅ Per-story folder |
| Config JSON | `./story_configs_batch_{N}_{timestamp}.json` | ✅ Timestamped |
| Analytics JSON | `./generation_analytics_batch_{N}_{timestamp}.json` | ✅ Timestamped |
| API call logs | `./logs/api_calls.jsonl` | ✅ Appends (cumulative) |
| Response cache | `./.cache/` | ✅ Prevents duplicate API calls |

---

## Models Used

| Purpose | Model | Cost (per 1M tokens) |
|---------|-------|---------------------|
| Story text generation | `gemini-2.5-flash` | $0.075 input / $0.30 output |
| Image generation | `gemini-2.5-flash-image` | Free tier available |

**Pro model (`gemini-1.5-pro`) is currently DISABLED** for cost efficiency.

---

## Safety Features

- **Prohibited content**: Violence, death, frightening imagery, romantic content, stereotypes, etc.
- **Silent reframe**: Unsafe prompts transformed without refusal ("zombie attack" → "sleepy neighbors")
- **Character safety**: Names never appear in image prompts (only physical traits)
- **Visual consistency**: 3-trait character description locked across all page images
- **Image safety**: All prompts end with no text, no words, no letters

---

## Cost Estimates

| Batch Size | Text Cost | Image Cost | Total Time |
|------------|-----------|------------|------------|
| 1 story | ~$0.001 | Free tier | ~2-5 min |
| 10 stories | ~$0.01 | Free tier | ~20-50 min |
| 100 stories | ~$0.12 | Free tier | ~3-8 hours |

---

## Commands

```bash
python generate.py 1                # 1 story with images + Word doc
python generate.py 5 --text-only    # 5 stories, JSON only (fast)
python generate.py 1 --dry-run      # Mock data, no API calls
python analytics.py                 # Cost breakdown from logs
python diagnostics.py               # Stress test rate limiter + concurrency
python smoke_test.py                # Verify all 6 modules
```

---

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `GEMINI_API_KEY` | ✅ | — | Google AI API key |
| `IMAGE_DELAY_SECONDS` | ❌ | 4 | Seconds between image API calls |
