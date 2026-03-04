#!/usr/bin/env bash
#
# Download Mercury and Gemini transcript files from the Spacelog GitHub repo.
# These are pre-processed text transcripts, not HTML scraping.
#
# Skips: Apollo missions (already covered by AFJ/ALSJ), Vostok, and
#        PAO/MEDIA transcripts (press commentary, not crew dialogue).
#        Also skips g7 which has no transcript files in the repo.
#
# Output: data/raw/Spacelog/{mission}/transcript.txt

set -euo pipefail

BASE="https://raw.githubusercontent.com/Spacelog/Spacelog/master/missions"
OUT_ROOT="data/raw/Spacelog"

# mission_dir : display_name : transcript_file
MISSIONS=(
    "mr3:Mercury-Redstone 3:ATG"
    "mr4:Mercury-Redstone 4:ATG"
    "ma6:Mercury-Atlas 6:TEC"
    "ma7:Mercury-Atlas 7:TEC"
    "ma8:Mercury-Atlas 8:TEC"
    "g3:Gemini 3:TEC"
    "g4:Gemini 4:TEC"
    "g6:Gemini 6:TEC"
    "g8:Gemini 8:TEC"
)

count=${#MISSIONS[@]}
i=0

for entry in "${MISSIONS[@]}"; do
    IFS=':' read -r mission_dir display_name transcript_file <<< "$entry"
    i=$((i + 1))
    out_dir="${OUT_ROOT}/${display_name}"
    mkdir -p "$out_dir"

    if [[ -f "${out_dir}/transcript.txt" ]]; then
        echo "Skipping (${i}/${count}): ${display_name}"
        continue
    fi

    echo "Downloading (${i}/${count}): ${display_name} (${transcript_file})"
    curl -s -o "${out_dir}/transcript.txt" \
        "${BASE}/${mission_dir}/transcripts/${transcript_file}"
    sleep 1
done

echo "Done. Downloaded ${count} mission transcripts."
