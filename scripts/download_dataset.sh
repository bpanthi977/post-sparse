#!/usr/bin/env bash
set -eu

DATA_DIR="${1:-data/coco}"
mkdir -p "$DATA_DIR"

download_and_extract() {
    local url="$1"
    local zip="$DATA_DIR/$(basename "$url")"
    local marker="$2"

    if [ -e "$DATA_DIR/$marker" ]; then
        echo "Skipping $(basename "$url") — already present."
        return
    fi

    echo "Downloading $(basename "$url") ..."
    wget --progress=bar:force -O "$zip" "$url"

    echo "Extracting $(basename "$zip") ..."
    unzip -q "$zip" -d "$DATA_DIR"
    rm "$zip"
    echo "Done."
}

download_and_extract \
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
    "annotations/captions_train2017.json"

download_and_extract \
    "http://images.cocodataset.org/zips/val2017.zip" \
    "val2017"

download_and_extract \
    "http://images.cocodataset.org/zips/train2017.zip" \
    "train2017"

echo ""
echo "Dataset ready at $DATA_DIR:"
for marker in "annotations/captions_train2017.json" "val2017" "train2017"; do
    if [ -e "$DATA_DIR/$marker" ]; then
        echo "  [OK]      $DATA_DIR/$marker"
    else
        echo "  [MISSING] $DATA_DIR/$marker"
    fi
done
