#!/bin/bash
# Downloads and organizes the DIV2K dataset (HR + LR bicubic X4)
# Usage: ./download_div2k.sh [destination_folder]
# Default: data/raw/DIV2K (relative to wherever you run this script from)

set -e  # stop the script if any command fails

DEST="${1:-data/raw/DIV2K}"
BASE_URL="https://data.vision.ee.ethz.ch/cvl/DIV2K"
SCALE="X4"

echo "Destination: $DEST"
mkdir -p "$DEST"
cd "$DEST"

FILES=(
  "DIV2K_train_HR.zip"
  "DIV2K_valid_HR.zip"
  "DIV2K_train_LR_bicubic_${SCALE}.zip"
  "DIV2K_valid_LR_bicubic_${SCALE}.zip"
)

for FILE in "${FILES[@]}"; do
  if [ -f "$FILE" ]; then
    echo "[OK] $FILE already exists, skipping download."
  else
    echo "[downloading] $FILE"
    wget -q --show-progress "${BASE_URL}/${FILE}"
  fi
done

echo "Extracting files..."
for FILE in "${FILES[@]}"; do
  echo "  -> $FILE"
  unzip -q -o "$FILE"
done

echo ""
echo "Done. Folder structure in $DEST:"
find "$DEST" -maxdepth 2 -type d | sort

echo ""
echo "Tip: to save disk space, you can delete the .zip files after confirming extraction went well:"
echo "  rm $DEST/*.zip"