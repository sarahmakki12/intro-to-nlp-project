# Project Notes: Data Collection and Preprocessing

## Goal

Build a character-level autocomplete system for astronaut communications across multiple languages. The model predicts the next character given a string context. The training data must reflect the domain: technical dialogue between astronauts and mission control, with a focus on **astronaut-side utterances only**, since the system is designed for what astronauts would type.

## Why Astronaut-Only Lines

The assignment specifies that the autocomplete system is for astronauts — it predicts what an astronaut would say next. Ground control (CapCom), PAO (Public Affairs Officer), and other personnel have different vocabularies and communication patterns. Including their lines would dilute the model with language patterns the astronaut would never use (e.g., "Flight, we have a go for..."). We filter strictly to crew members for every corpus.

## Corpora

We use three corpora of real NASA mission transcripts, chosen for their density of technical astronaut dialogue:

### 1. Apollo Flight Journal (AFJ)

- **Source**: https://apollojournals.org/afj/
- **Missions**: Apollo 7 through 17 (all crewed Apollo missions)
- **Content**: Full mission transcripts covering launch through splashdown, including in-flight dialogue, procedures, and status reports
- **Why**: The AFJ provides the most comprehensive record of astronaut communications during transit phases. These transcripts are rich in technical vocabulary (orbital mechanics, navigation, systems status) and represent the bulk of our training data.
- **Lines extracted**: ~120,633

### 2. Apollo Lunar Surface Journal (ALSJ)

- **Source**: https://apollojournals.org/alsj/
- **Missions**: Apollo 11, 12, 14, 15, 16, 17 (missions with lunar landings; Apollo 13 had no landing)
- **Content**: Transcripts of lunar surface operations — EVAs, geology observations, equipment discussions
- **Why**: Complements the AFJ with surface-specific vocabulary (geology terms, EVA procedures, lunar module operations) that doesn't appear in the flight transcripts. The ALSJ covers a different communication context (surface ops vs. transit) which adds variety.
- **Lines extracted**: ~44,503

### 3. Spacelog (Mercury and Gemini)

- **Source**: https://github.com/Spacelog/Spacelog (raw transcript files from the GitHub repo)
- **Missions**: Mercury-Redstone 3 & 4, Mercury-Atlas 6, 7, & 8, Gemini 3, 4, 6, & 8
- **Content**: Early NASA program transcripts predating Apollo
- **Why**: Adds linguistic variety from a different era of spaceflight. Mercury and Gemini missions have shorter, more terse communications and different technical vocabulary (capsule systems vs. Apollo's command/service module). Apollo missions were excluded from this source since they're already covered by AFJ/ALSJ. Vostok was excluded as out of scope for now.
- **Lines extracted**: ~13,147

### Combined Total: ~178,283 lines in `data/training/en.txt`

## Data Pipeline

### Directory Structure

```
src/data/
├── scraping/
│   ├── apollo_flight_journal.sh
│   ├── apollo_lunar_surface_journal.sh
│   └── spacelog.sh
├── cleaning/
│   ├── apollo_flight_journal.py
│   ├── apollo_lunar_surface_journal.py
│   └── spacelog.py
├── combine_cleaned.py
├── dedup.py
└── translate.py

data/
├── raw/              # Downloaded HTML/text files (not committed)
├── cleaned/          # Per-corpus cleaned text files
└── training/
    ├── en.txt        # Combined English training data (178K lines)
    ├── en_capped.txt # English with duplicates capped at 10 (150K lines)
    ├── en_unique.txt # Unique English lines (138K lines, translation source)
    └── {lang}.txt    # Translated training data (ru, zh, ja, hi, ar, ko, fr, de, it)
```

Each corpus has its own scraping and cleaning script because the source formats differ significantly. The combine script is general-purpose and merges everything.

### Step 1: Scraping

All scraping scripts include a file-existence check to avoid re-downloading and a progress counter.

**Apollo Flight Journal:**
```bash
bash src/data/scraping/apollo_flight_journal.sh
```
- Fetches each mission's `index.html`, extracts links to transcript pages via `grep`
- Filters out non-transcript pages (photos, videos, bibliographies, reference docs) via an exclude pattern
- Filters out absolute URLs and subdirectory links (`grep -v '/'`)
- Downloads each page with `curl`, 1-second delay between requests

**Apollo Lunar Surface Journal:**
```bash
bash src/data/scraping/apollo_lunar_surface_journal.sh
```
- Fetches each mission's main page, extracts links from the "Journal" section (between `<h2>The Journal</h2>` and the next `<h2>`)
- Filters to local HTML files only (no AFJ cross-links or external URLs)
- Same download pattern as AFJ

**Spacelog:**
```bash
bash src/data/scraping/spacelog.sh
```
- Downloads raw transcript text files directly from the Spacelog GitHub repository (not from spacelog.org, which has performance issues and requires paginated loading)
- Each mission's transcript is a single text file, so no index-page parsing is needed
- Mission list and transcript filenames are hardcoded since the GitHub repo structure is known

### Step 2: Cleaning

Each cleaning script is a Python script that reads raw files and produces cleaned text files with one dialogue line per line.

```bash
python src/data/cleaning/apollo_flight_journal.py
python src/data/cleaning/apollo_lunar_surface_journal.py
python src/data/cleaning/spacelog.py
```

#### Astronaut Filtering

Each script maintains an explicit mapping of crew members per mission:

- **AFJ**: Crew names (e.g., "Armstrong", "Collins", "Aldrin" for Apollo 11) plus role labels ("CDR", "CMP", "LMP", "SC", "Spacecraft")
- **ALSJ**: Crew names per mission plus "LM Crew" as a group label
- **Spacelog**: Speaker codes per mission (e.g., "P" for pilot in Mercury missions, "C"/"P" for commander/pilot in Gemini). These were determined by examining the `_meta` files in the Spacelog repository.

Only lines from these speakers are kept. All ground control, PAO, and other personnel are discarded.

#### HTML/Format Parsing

Each corpus has a different raw format:

- **AFJ**: HTML with `<div class="cc">` blocks containing `TIMESTAMP SPEAKER: dialogue` patterns
- **ALSJ**: HTML with `<b>TIMESTAMP</b> Speaker: dialogue text<p>` patterns, timestamps in `HHH:MM:SS` or `HHH:MM:xx` format
- **Spacelog**: Plain text with `[TIMESTAMP]\nSPEAKER: dialogue` format, plus `_page`/`_tape` metadata lines and `[glossary:...]` tags

#### Text Cleaning Pipeline

After extracting the raw dialogue text, all three scripts apply the same cleaning pipeline:

1. **Strip HTML tags**: Remove all remaining HTML (`<sub>`, `<a>`, `<b>`, etc.) and unescape HTML entities
2. **Drop garbled lines**: Any line containing "garble" (case-insensitive) is dropped entirely, since these represent unintelligible audio that would be noise for training
3. **Unwrap fill-in parentheticals**: Parenthetical content consisting of 1-3 common English words (e.g., `(I)`, `(the)`, `(we are)`) has the parentheses removed but the words kept — so `"Yes, (I) think so"` becomes `"Yes, I think so"`
4. **Strip remaining parentheticals**: All other `(...)` content is removed entirely — this catches stage directions like `(Long Pause)`, `(laughter)`, timing annotations, etc.
5. **Strip bracketed annotations**: All `[...]` content is removed — these are editorial notes like `[garble, probably tower]` (though lines with "garble" are already dropped in step 2)
6. **Clean up unclosed delimiters**: Remove content from any unclosed `(` or `[` to end of line
7. **Remove stray delimiter characters**: Final pass to strip any orphaned `(){}[]` characters
8. **Whitespace normalization**: Collapse multiple spaces, trim
9. **Minimum length filter**: Lines shorter than 2 characters are dropped

#### Design Decision: Keeping Fill-In Words

The original transcripts contain parenthetical "fill-in words" where transcribers inferred what the astronaut probably said through static or garble — e.g., `"Yes, (I) think that's right"`. We initially dropped these lines entirely to avoid training on potentially incorrect guesses.

We later reconsidered: since the purpose of the model is autocomplete (predicting the next character an astronaut would type), these fill-in words are exactly the kind of common words the model should learn to predict. The transcribers' guesses for these short, common words (pronouns, articles, conjunctions) are very likely correct, and stripping them would leave grammatically broken sentences that are worse for training. We now unwrap them — keeping the word, removing only the parentheses — while still stripping all other parenthetical annotations (pauses, stage directions, etc.) that are not part of the actual dialogue.

### Step 3: Combining

```bash
python src/data/combine_cleaned.py
```

Recursively collects all `.txt` files under `data/cleaned/` and writes them to `data/training/en.txt`, one line per line. This is corpus-agnostic — adding a new corpus only requires placing cleaned `.txt` files in a subdirectory of `data/cleaned/`.

## Running the Full Pipeline

```bash
# 1. Download raw data
bash src/data/scraping/apollo_flight_journal.sh
bash src/data/scraping/apollo_lunar_surface_journal.sh
bash src/data/scraping/spacelog.sh

# 2. Clean each corpus
python src/data/cleaning/apollo_flight_journal.py
python src/data/cleaning/apollo_lunar_surface_journal.py
python src/data/cleaning/spacelog.py

# 3. Combine into training file
python src/data/combine_cleaned.py
```

Output: `data/training/en.txt` (~178,283 lines of clean English astronaut dialogue)

### Step 4: Deduplication / Capping

```bash
python src/data/dedup.py
```

The raw combined corpus contains significant repetition — common acknowledgments like "Okay." (8,342 occurrences), "Roger." (2,240), "Yes." (2,731) dominate. While these are legitimate dialogue patterns the model should learn, having thousands of copies wastes translation budget and skews training.

We cap each unique line at 10 occurrences maximum, preserving original order. This produces:

- **`en_capped.txt`** (~150,018 lines): for English training — retains frequency signal for common phrases without extreme repetition
- **`en_unique.txt`** (~138,043 lines): strictly unique lines in first-occurrence order, used as the source for translation

| Metric | Count |
|--------|-------|
| Original lines | 178,283 |
| After capping (≤10) | 150,018 |
| Unique lines | 138,043 |
| Duplicates removed | 28,265 (15.9%) |

### Step 5: Translation

```bash
python src/data/translate.py
```

Translates `en_unique.txt` into the 9 target languages (ru, zh, ja, hi, ar, ko, fr, de, it) using Google Translate via the `deep-translator` Python library. Key features:

- **Batching**: Lines are joined with `\n` into batches of ~4,500 characters per API request, reducing the number of calls ~70x compared to line-by-line translation. If Google merges/splits lines in a batch, the script falls back to line-by-line for that batch.
- **Rate limiting**: Configurable delay between requests (default 0.5s) with exponential backoff on errors (HTTP 429, 500, etc.).
- **Checkpoint/resume**: Progress is saved per language after each batch. If interrupted, re-running the script picks up where it left off.

Output: `data/training/{lang}.txt` for each target language, with one translated line per source line in the same order as `en_unique.txt`.

#### Running translation in the background

The full translation takes ~5 hours (138K lines x 9 languages). Run it via `nohup` so it persists across terminal/session closures:

```bash
cd /home/sarahmakki12/cs489/intro-to-nlp-project
nohup .venv/bin/python3 src/data/translate.py > data/training/translate.log 2>&1 &
```

To monitor progress:

```bash
# Check which line the current language is on
cat data/training/.progress_ru   # (or .progress_zh, .progress_ja, etc.)

# See which languages have progress files (started or completed)
ls data/training/.progress_*

# Check if the process is still alive (use PID from launch output, or find it)
ps -p <PID>
ps aux | grep translate.py | grep -v grep
```

If interrupted, re-run the same `nohup` command -- checkpoint/resume skips already-translated lines.

To reset a language and re-translate from scratch, delete its output and progress files:

```bash
rm data/training/ru.txt data/training/.progress_ru
```

## Running the Full Pipeline

```bash
# 1. Download raw data
bash src/data/scraping/apollo_flight_journal.sh
bash src/data/scraping/apollo_lunar_surface_journal.sh
bash src/data/scraping/spacelog.sh

# 2. Clean each corpus
python src/data/cleaning/apollo_flight_journal.py
python src/data/cleaning/apollo_lunar_surface_journal.py
python src/data/cleaning/spacelog.py

# 3. Combine into training file
python src/data/combine_cleaned.py

# 4. Deduplicate / cap
python src/data/dedup.py

# 5. Translate (long-running, supports resume)
python src/data/translate.py
```

## Next Steps

### In progress

- **Translation running in background** — translating `en_unique.txt` into 9 languages via Google Translate. Monitor with `cat data/training/.progress_*` and `ps aux | grep translate.py`.

### After translation completes

- Verify line counts match for each language: every `data/training/{lang}.txt` should have 138,043 lines (same as `en_unique.txt`)
- Check for untranslated lines (compare each translated line to its English source, report stats per language)
- Reassemble capped multilingual training files by replaying the cap-at-10 logic from `en.txt` onto each translated language file (map each English line to its translation, preserve the same repetition pattern)

### Restructure training data format

The current pipeline outputs flat `.txt` files with no provenance tracking. This should be refactored into a structured CSV format:

1. **Redo `combine_cleaned.py`** to produce a CSV with columns: `text`, `corpus` (AFJ / ALSJ / Spacelog), `mission`, `file`, `line_number`. This preserves the source of every line.

2. **Redo `dedup.py`** to read this CSV and produce a deduplicated CSV with columns: `unique_id`, `text`, `corpus`, `mission`, `file`, `line_number`, `occurrence_count`. The `unique_id` is assigned per unique text, `corpus`/`mission`/`file`/`line_number` are from the first occurrence, and `occurrence_count` tracks how many times the line appeared (capped or uncapped).

3. **After translation**, each `{lang}.txt` has lines in the same order as `en_unique.txt`. Build a final combined multilingual CSV with columns: `id` (row number), `unique_id` (links translations of the same sentence), `lang`, `text`, `corpus`, `mission`, `file`, `line_number`, `occurrence_count`. This means every English sentence and its 9 translations share the same `unique_id`, making it easy to cross-reference or filter by language.

### Model development

- Build baseline n-gram model using kaggle training data (`data/kaggle/train.csv`) for immediate iteration
- Evaluate on `data/open-dev/input.txt`
- Incorporate translated corpus once available to improve non-English performance
- Explore more advanced models (transformer, fine-tuned/distilled pretrained model) if n-gram baseline meets midterm threshold

### Optional: additional native-language data

- Explore transcripts from non-American space agencies for additional variety, especially for non-English languages:
  - **ESA** (European Space Agency): mission transcripts and astronaut communications, potentially useful for French, German, and Italian
  - **Roscosmos / Soviet missions**: Vostok, Voskhod, Soyuz, and Mir transcripts for Russian-language data
  - **JAXA** (Japan Aerospace Exploration Agency): Japanese astronaut communications from ISS expeditions
  - **ISRO** (Indian Space Research Organisation): Gaganyaan and other mission communications for Hindi
  - **KARI** (Korea Aerospace Research Institute): Korean astronaut communications for Korean-language data
  - These could provide native-language technical dialogue rather than relying solely on machine translation of English transcripts
