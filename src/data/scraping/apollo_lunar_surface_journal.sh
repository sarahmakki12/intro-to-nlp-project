#!/usr/bin/env bash
#
# Download Apollo Lunar Surface Journal transcript HTML files (Apollo 11–17).
# Apollo 13 had no lunar landing and is not included in the ALSJ.
#
# For each mission, fetches the index page, extracts links from the "Journal"
# section (between <h2>The Journal</h2> and the next <h2>), keeping only local
# HTML files (no parent/external links).
#
# Output: data/raw/Apollo Lunar Surface Journal/Apollo {N}/*.html

set -euo pipefail

BASE="https://apollojournals.org/alsj"
OUT_ROOT="data/raw/Apollo Lunar Surface Journal"

MISSIONS=(a11 a12 a14 a15 a16 a17)

for mission in "${MISSIONS[@]}"; do
    mission_num=$((10#$(echo "$mission" | grep -oP '[0-9]+')))
    out_dir="${OUT_ROOT}/Apollo ${mission_num}"
    mkdir -p "$out_dir"

    echo "=== Apollo ${mission_num} ==="

    links=$(curl -s "${BASE}/${mission}/${mission}.html" \
        | sed -n '/<h2>The Journal<\/h2>/,/<h2>/p' \
        | grep -oP 'href="\K[^"]*\.html' \
        | grep -v '/')

    count=$(echo "$links" | wc -l)
    echo "  Found ${count} journal pages"

    i=0
    for page in $links; do
        i=$((i + 1))
        if [[ -f "${out_dir}/${page}" ]]; then
            echo "  Skipping (${i}/${count}): ${page}"
            continue
        fi
        echo "  Downloading (${i}/${count}): ${page}"
        curl -s -o "${out_dir}/${page}" "${BASE}/${mission}/${page}"
        sleep 1
    done

    echo "  Done."
    echo
done

echo "All missions downloaded."
