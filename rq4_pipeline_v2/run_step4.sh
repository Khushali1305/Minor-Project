#!/bin/bash
set -euo pipefail

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 4: Run Models + Score + Judge                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

INPUT="output"
OUTPUT="output"

# ── Phase 4a: Run models as simplifiers ──
echo "═══ Phase 4a: Model inference ═══"
echo ""

# Local GPU models (fastest)
for M in "flan-t5-large" "flan-t5-xl"; do
    echo "── $M (local GPU) ──"
    python3 step4_run_and_score.py --input-dir $INPUT --output-dir $OUTPUT --phase run --models "$M" --delay 0
    echo ""
done

# Ollama models
for M in "llama-3.1-8b" "qwen2.5-32b" "gemma-3-27b"; do
    echo "── $M (ollama) ──"
    python3 step4_run_and_score.py --input-dir $INPUT --output-dir $OUTPUT --phase run --models "$M" --delay 0.1
    echo ""
done

# API models
echo "── llama-3.1-70b (sambanova) ──"
python3 step4_run_and_score.py --input-dir $INPUT --output-dir $OUTPUT --phase run --models "llama-3.1-70b" --delay 2
echo ""

echo "── gpt-4.1 (openai) ──"
python3 step4_run_and_score.py --input-dir $INPUT --output-dir $OUTPUT --phase run --models "gpt-4.1" --delay 0.3
echo ""

# ── Phase 4b: Rule-based scoring ──
echo "═══ Phase 4b: Rule-based scoring ═══"
python3 step4_run_and_score.py --input-dir $INPUT --output-dir $OUTPUT --phase score
echo ""

# ── Phase 4c: LLM Judges ──
echo "═══ Phase 4c: LLM Judges (Claude Haiku + Mistral Large 3) ═══"
python3 step4_run_and_score.py --input-dir $INPUT --output-dir $OUTPUT --phase judge --delay 0.3
echo ""

# ── Phase 4d: Summary ──
echo "═══ Phase 4d: Final reliability summary ═══"
python3 step4_run_and_score.py --input-dir $INPUT --output-dir $OUTPUT --phase summary
echo ""

echo "✓ Step 4 complete. Check output/simp_scores/reliability_summary.csv"
