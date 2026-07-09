#!/bin/bash
# scripts/download_data_Wang_2023.sh
#
# Downloads the 3 files needed from the Zenodo deposit associated with
# Wang et al. (2023), DOI: 10.5281/zenodo.7470938, saves them to
# data/raw/Wang_2023/, and prints the dimensions of all three.
#
# Usage (from the super-resolution/ repo root):
#   ./scripts/download_data_Wang_2023.sh

set -e

RECORD_ID="7470938"
RAW_DIR="data/raw/Wang_2023"
mkdir -p "$RAW_DIR"

FILES_NEEDED=("ffov_crop_origsize.tiff" "PEFC_hres_0p7um.tiff" "LRTest.tif")

echo "Fetching file list from Zenodo record ${RECORD_ID}..."
METADATA=$(curl -s "https://zenodo.org/api/records/${RECORD_ID}")

if [ -z "$METADATA" ] || ! echo "$METADATA" | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
    echo "ERROR: could not reach Zenodo or received an invalid response."
    echo "Check your network connection and try again."
    exit 1
fi

for FNAME in "${FILES_NEEDED[@]}"; do
    URL=$(echo "$METADATA" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for f in data['files']:
    if f['key'] == '$FNAME':
        print(f['links']['self'])
        break
")
    if [ -z "$URL" ]; then
        echo "WARNING: could not find download URL for $FNAME -- check the record ID or filename."
        continue
    fi

    if [ -f "${RAW_DIR}/${FNAME}" ]; then
        echo "Skipping $FNAME (already exists in ${RAW_DIR}/)"
        continue
    fi

    echo "Downloading $FNAME..."
    wget -q --show-progress -O "${RAW_DIR}/${FNAME}" "$URL"
done

echo ""
echo "Files saved to ${RAW_DIR}/"
echo ""
echo "Dimensions:"
python3 -c "
import tifffile
from pathlib import Path

raw_dir = Path('${RAW_DIR}')
for fname in ['ffov_crop_origsize.tiff', 'PEFC_hres_0p7um.tiff', 'LRTest.tif']:
    path = raw_dir / fname
    if not path.exists():
        print(f'  {fname}: NOT FOUND')
        continue
    array = tifffile.imread(str(path))
    print(f'  {fname}: shape = {array.shape}, dtype = {array.dtype}')
"
