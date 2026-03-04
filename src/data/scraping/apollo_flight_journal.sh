#!/usr/bin/env bash
#
# Download Apollo Flight Journal transcript HTML files (Apollo 7–17).
#
# For each mission, fetches the index page, extracts links to transcript pages
# (filtering out documents, photo/video indexes, bibliographies, etc.),
# and downloads only those pages.
#
# Output: data/raw/Apollo Flight Journal/Apollo {N}/*.html

set -euo pipefail

BASE="https://apollojournals.org/afj"
OUT_ROOT="data/raw/Apollo Flight Journal"
EXCLUDE_PATTERN='(document|photo|video|update|biblio|explan|image|summary|index|audio|graffiti|weather|lightningstrike|essay|reference|a15docs|aoh_op_procs|transcription)'

MISSIONS=(ap07fj ap08fj ap09fj ap10fj ap11fj ap12fj ap13fj ap14fj ap15fj ap16fj ap17fj)

for mission_path in "${MISSIONS[@]}"; do
    mission_num=$((10#$(echo "$mission_path" | grep -oP '[0-9]+')))
    out_dir="${OUT_ROOT}/Apollo ${mission_num}"
    mkdir -p "$out_dir"

    echo "=== Apollo ${mission_num} ==="

    links=$(curl -s "${BASE}/${mission_path}/index.html" \
        | grep -oP 'href="\K[^"]*\.html' \
        | grep -v '\.\./' \
        | grep -v '/' \
        | grep -v -iE "$EXCLUDE_PATTERN")

    count=$(echo "$links" | wc -l)
    echo "  Found ${count} transcript pages"

    for page in $links; do
        if [[ -f "${out_dir}/${page}" ]]; then
            echo "  Skipping (exists): ${page}"
            continue
        fi
        echo "  Downloading: ${page}"
        curl -s -o "${out_dir}/${page}" "${BASE}/${mission_path}/${page}"
        sleep 1
    done

    echo "  Done."
    echo
done

echo "All missions downloaded."
