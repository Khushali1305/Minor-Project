#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════════════════
#  RQ4 PIPELINE — Readability-Controlled Simplification
#  Target: Lightning AI (Linux + NVIDIA GPU)
# ═══════════════════════════════════════════════════════════════════════════
#
#  Timeline (24 hours):
#    Phase A  [auto]   Steps 1-2 (rule-based probes)           ~10 min
#    Phase B  [auto]   Step 2   (semantic probes via models)   ~3-6 hrs
#    Phase C  [auto]   Step 3   (audit sample)                 ~1 min
#    ── MANUAL STOP: audit 120 probes in CSV ──                ~2-3 hrs
#    Phase D  [auto]   Step 3   (freeze)                       ~1 min
#    Phase E  [auto]   Step 4   (run models + score + judge)   ~6-10 hrs
#    Phase F  [auto]   Step 5   (cross-domain analysis)        ~1 min
#
#  API keys needed (set before running):
#    export OPENAI_API_KEY="sk-..."          # for GPT-4.1
#    export GROQ_API_KEY="gsk_..."           # for Llama-3.1-70B
#    export ANTHROPIC_API_KEY="sk-ant-..."   # for Claude Haiku judge
#    export MISTRAL_API_KEY="..."            # for Mistral Large judge
#
#  Data files needed in ./data/:
#    tsar2025_train.json
#    onestop_english_train.json
#    onestop_english_test.json
#    asset_simplification_test.json
#    asset_simplification_validation.json
# ═══════════════════════════════════════════════════════════════════════════

DATA_DIR="data"
OUTPUT_DIR="output"
LOG_DIR="logs"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           RQ4 PIPELINE — Lightning AI GPU                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ═══ ENVIRONMENT SETUP ═══
echo "[SETUP] Checking environment..."

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "  CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    echo "  WARNING: No GPU detected. Local models will run on CPU (slow)."
fi

# Python packages
echo "[SETUP] Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt 2>&1 | tail -3

# NLTK data (for optional WordNet synonyms)
python -c "
try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print('  NLTK data: OK')
except: print('  NLTK data: skipped (using built-in synonyms)')
" 2>/dev/null || true

# Ollama setup (for local models: llama3.1, qwen2.5, gemma3)
echo "[SETUP] Checking ollama..."
if ! command -v ollama &> /dev/null; then
    echo "  Installing ollama..."
    curl -fsSL https://ollama.com/install.sh | sh 2>&1 | tail -3
fi

# Start ollama server if not running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  Starting ollama server..."
    ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    sleep 5
fi

# Pull required models (runs in background, continues while pulling)
echo "[SETUP] Pulling ollama models (background)..."
for MODEL in "llama3.1:8b-instruct-q8_0" "qwen2.5:32b" "gemma3:27b"; do
    if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
        echo "  Pulling $MODEL..."
        ollama pull "$MODEL" > "$LOG_DIR/pull_${MODEL//[:\/]/_}.log" 2>&1 &
    else
        echo "  $MODEL: already available"
    fi
done

# Check API keys
echo ""
echo "[SETUP] API keys:"
[ -n "$OPENAI_API_KEY" ]    && echo "  OPENAI_API_KEY: set"    || echo "  OPENAI_API_KEY: MISSING (GPT-4.1 will fail)"
[ -n "$GROQ_API_KEY" ]      && echo "  GROQ_API_KEY: set"      || echo "  GROQ_API_KEY: MISSING (Llama-70B will fail)"
[ -n "$ANTHROPIC_API_KEY" ] && echo "  ANTHROPIC_API_KEY: set"  || echo "  ANTHROPIC_API_KEY: MISSING (Claude judge will fail)"
[ -n "$MISTRAL_API_KEY" ]   && echo "  MISTRAL_API_KEY: set"    || echo "  MISTRAL_API_KEY: MISSING (Mistral judge will fail)"

# Check data files
echo ""
echo "[SETUP] Data files:"
MISSING=0
for F in tsar2025_train.json onestop_english_train.json onestop_english_test.json asset_simplification_test.json asset_simplification_validation.json; do
    if [ -f "$DATA_DIR/$F" ]; then
        echo "  $F: OK"
    else
        echo "  $F: MISSING"
        MISSING=1
    fi
done
if [ $MISSING -eq 1 ]; then
    echo ""
    echo "ERROR: Missing data files in $DATA_DIR/. Place them there and re-run."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " PHASE A: Normalize Data (Step 1)"
echo "═══════════════════════════════════════════════════════════"
echo ""

if [ -f "$OUTPUT_DIR/simp_master_table.jsonl" ]; then
    echo "  [SKIP] Master table already exists."
else
    python step1_normalize.py --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/step1.log"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " PHASE A: Generate Rule-Based Probes (Step 2, no semantic)"
echo "═══════════════════════════════════════════════════════════"
echo ""

python step2_generate_probes.py \
    --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR" \
    --skip-semantic 2>&1 | tee "$LOG_DIR/step2_rule.log"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " PHASE B: Generate Semantic Probes (Step 2, per model)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Wait for ollama models to finish pulling
echo "  Waiting for ollama model pulls to complete..."
wait

# Run semantic probes model by model (resumable)
# Local GPU models first (fastest on Lightning AI)
for MODEL in "flan-t5-large" "flan-t5-xl"; do
    echo ""
    echo "  ── Semantic probes: $MODEL (local GPU) ──"
    python step2_generate_probes.py \
        --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR" \
        --models "$MODEL" --delay 0 2>&1 | tee -a "$LOG_DIR/step2_semantic.log"
done

# Ollama models
for MODEL in "llama-3.1-8b" "qwen2.5-32b" "gemma-3-27b"; do
    echo ""
    echo "  ── Semantic probes: $MODEL (ollama) ──"
    python step2_generate_probes.py \
        --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR" \
        --models "$MODEL" --delay 0.1 2>&1 | tee -a "$LOG_DIR/step2_semantic.log"
done

# API models
for MODEL in "llama-3.1-70b" "gpt-4.1" "gpt-4.1-mini"; do
    echo ""
    echo "  ── Semantic probes: $MODEL (API) ──"
    python step2_generate_probes.py \
        --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR" \
        --models "$MODEL" --delay 0.3 2>&1 | tee -a "$LOG_DIR/step2_semantic.log"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " PHASE C: Generate Audit Sample (Step 3)"
echo "═══════════════════════════════════════════════════════════"
echo ""

python step3_audit.py --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR" --mode sample \
    2>&1 | tee "$LOG_DIR/step3_sample.log"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║  ██  MANUAL STOP — AUDIT 120 PROBES  ██                 ║"
echo "║                                                          ║"
echo "║  1. Open:  $OUTPUT_DIR/simp_audit_form.csv              ║"
echo "║  2. Mark VERDICT column: KEEP / FIX / REJECT            ║"
echo "║  3. Save as: $OUTPUT_DIR/simp_audit_completed.csv       ║"
echo "║  4. Press ENTER here to continue                         ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
read -p "Press ENTER after completing the audit... "

# Verify audit file exists
if [ ! -f "$OUTPUT_DIR/simp_audit_completed.csv" ]; then
    echo "ERROR: $OUTPUT_DIR/simp_audit_completed.csv not found!"
    echo "Complete the audit and re-run from this point."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " PHASE D: Freeze Simp-ProbeCore v1 (Step 3)"
echo "═══════════════════════════════════════════════════════════"
echo ""

python step3_audit.py --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR" --mode freeze \
    2>&1 | tee "$LOG_DIR/step3_freeze.log"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " PHASE E: Run Models + Score + Judge (Step 4)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Run models — local GPU first, then ollama, then API
echo "── Running local GPU models ──"
for MODEL in "flan-t5-large" "flan-t5-xl"; do
    python step4_run_and_score.py \
        --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR" \
        --phase run --models "$MODEL" --delay 0 2>&1 | tee -a "$LOG_DIR/step4_run.log"
done

echo "── Running ollama models ──"
for MODEL in "llama-3.1-8b" "qwen2.5-32b" "gemma-3-27b"; do
    python step4_run_and_score.py \
        --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR" \
        --phase run --models "$MODEL" --delay 0.1 2>&1 | tee -a "$LOG_DIR/step4_run.log"
done

echo "── Running API models ──"
for MODEL in "llama-3.1-70b" "gpt-4.1" "gpt-4.1-mini"; do
    python step4_run_and_score.py \
        --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR" \
        --phase run --models "$MODEL" --delay 0.3 2>&1 | tee -a "$LOG_DIR/step4_run.log"
done

echo ""
echo "── Rule-based scoring ──"
python step4_run_and_score.py \
    --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR" \
    --phase score 2>&1 | tee "$LOG_DIR/step4_score.log"

echo ""
echo "── LLM Judges (Claude Haiku + Mistral Large 3) ──"
python step4_run_and_score.py \
    --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR" \
    --phase judge --delay 0.3 2>&1 | tee "$LOG_DIR/step4_judge.log"

echo ""
echo "── Final reliability summary ──"
python step4_run_and_score.py \
    --input-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR" \
    --phase summary 2>&1 | tee "$LOG_DIR/step4_summary.log"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " PHASE F: Cross-Domain Analysis (Step 5)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# If SRS scores exist, include them
SRS_FLAG=""
if [ -d "srs_scores" ]; then
    SRS_FLAG="--srs-dir srs_scores"
    echo "  SRS scores found — will generate cross-domain comparison"
fi

python step5_cross_domain.py \
    --simp-dir "$OUTPUT_DIR/simp_scores" \
    --output-dir "$OUTPUT_DIR/rq4_results" \
    $SRS_FLAG 2>&1 | tee "$LOG_DIR/step5.log"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║  ✓ RQ4 PIPELINE COMPLETE                                ║"
echo "║                                                          ║"
echo "║  Key outputs:                                            ║"
echo "║    $OUTPUT_DIR/simp_probecore_v1.jsonl   (frozen probes) ║"
echo "║    $OUTPUT_DIR/simp_scores/              (per-model)     ║"
echo "║    $OUTPUT_DIR/simp_judge_outputs/       (judge data)    ║"
echo "║    $OUTPUT_DIR/rq4_results/              (analysis)      ║"
echo "║                                                          ║"
echo "║  Paper tables:                                           ║"
echo "║    rq4_paper_table.csv                                   ║"
echo "║    rq4_cross_domain.csv  (if SRS scores provided)        ║"
echo "║    rq4_tier_comparison.csv                               ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Logs saved in: $LOG_DIR/"
