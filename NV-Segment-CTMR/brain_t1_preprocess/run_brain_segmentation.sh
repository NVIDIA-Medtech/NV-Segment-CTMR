#!/bin/bash

# Brain MRI Segmentation Script
# Supports single file or folder batch processing with optional temporary file retention

set -e  # Exit on error

# Default values
KEEP_TEMP=false
OUTPUT_DIR=""
MODALITY="MRI_BRAIN"
CONDA_ENV="vista3d-nv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Brain MRI Segmentation Script for NV-Segment-CTMR

OPTIONS:
    --input FILE              Single NIfTI file to segment
    --input_folder FOLDER     Folder containing NIfTI files to process (batch mode)
    --output_dir DIR          Output directory (default: ./eval)
    --keep-temp               Keep temporary preprocessing files (default: false)
    --modality MODALITY       Segmentation modality: MRI_BRAIN (default), MRI_BODY, CT_BODY
    --conda-env ENV           Conda environment name (default: vista3d-nv)
    -h, --help                Show this help message

EXAMPLES:
    # Single file processing
    $0 --input example/brain_t1.nii.gz

    # Batch processing
    $0 --input_folder example/ --output_dir results/

    # Keep temporary files for debugging
    $0 --input example/brain_t1.nii.gz --keep-temp

EOF
    exit 1
}

# Function to check if conda environment is activated
check_conda_env() {
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}Error: conda command not found. Please install conda or activate your environment manually.${NC}" >&2
        exit 1
    fi

    # Check if environment exists
    if ! conda env list | grep -q "^${CONDA_ENV} "; then
        echo -e "${YELLOW}Warning: Conda environment '${CONDA_ENV}' not found. Attempting to activate anyway...${NC}" >&2
    fi

    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}" || {
        echo -e "${RED}Error: Failed to activate conda environment '${CONDA_ENV}'.${NC}" >&2
        exit 1
    }

    echo -e "${GREEN}Activated conda environment: ${CONDA_ENV}${NC}"
}

# Function to process a single file
process_single_file() {
    local input_file="$1"
    local output_dir="${OUTPUT_DIR:-./eval}"

    if [[ ! -f "$input_file" ]]; then
        echo -e "${RED}Error: Input file not found: $input_file${NC}" >&2
        exit 1
    fi

    # Get absolute paths
    input_file=$(realpath "$input_file")
    output_dir=$(realpath -m "$output_dir")
    mkdir -p "$output_dir"

    # Extract filename without extension
    local file_basename=$(basename "$input_file" .nii.gz)
    file_basename=$(basename "$file_basename" .nii)
    local file_dir=$(dirname "$input_file")

    # Create temporary directory for this file
    local temp_dir="${file_dir}/${file_basename}_temp"
    mkdir -p "$temp_dir"

    # Temporary file paths
    local skull_stripped="${temp_dir}/${file_basename}_skull_stripped.nii.gz"
    local preprocess_tmp="${temp_dir}/${file_basename}_preprocessed.nii.gz"
    local preprocess_meta="${temp_dir}/${file_basename}_preprocessed.meta.json"
    local preprocess_tmp_seg="${output_dir}/${file_basename}_preprocessed/${file_basename}_preprocessed_trans.nii.gz"
    local final_output="${output_dir}/${file_basename}_trans.nii.gz"

    echo -e "${GREEN}Processing: $input_file${NC}"
    echo -e "${GREEN}Output will be saved to: $final_output${NC}"

    # Step 1: Skull stripping with SynthStrip
    echo -e "${YELLOW}Step 1/4: Skull stripping...${NC}"
    if [[ ! -f "$skull_stripped" ]]; then
        cd "$BUNDLE_ROOT"
        ./brain_t1_preprocess/synthstrip-docker -i "$input_file" -o "$skull_stripped" || {
            echo -e "${RED}Error: Skull stripping failed${NC}" >&2
            [[ "$KEEP_TEMP" == "false" ]] && rm -rf "$temp_dir"
            exit 1
        }
    else
        echo -e "${YELLOW}  Skull-stripped file already exists, skipping...${NC}"
    fi

    # Step 2: Affine align to the LUMIR template
    echo -e "${YELLOW}Step 2/4: Affine alignment to LUMIR template...${NC}"
    cd "$BUNDLE_ROOT"
    python brain_t1_preprocess/preprocess.py \
        "$skull_stripped" \
        brain_t1_preprocess/LUMIR_template.nii.gz \
        "$preprocess_tmp" \
        --save-preprocess "$preprocess_meta" || {
        echo -e "${RED}Error: Preprocessing failed${NC}" >&2
        [[ "$KEEP_TEMP" == "false" ]] && rm -rf "$temp_dir"
        exit 1
    }

    # Step 3: Segment the brain
    echo -e "${YELLOW}Step 3/4: Running segmentation...${NC}"
    cd "$BUNDLE_ROOT"
    python -m monai.bundle run \
        --config_file configs/inference.json \
        --input_dict "{'image':'$preprocess_tmp'}" \
        --modality "$MODALITY" || {
        echo -e "${RED}Error: Segmentation failed${NC}" >&2
        [[ "$KEEP_TEMP" == "false" ]] && rm -rf "$temp_dir"
        exit 1
    }

    # Step 4: Revert the segmentation back to original space
    echo -e "${YELLOW}Step 4/4: Reverting to original space...${NC}"
    if [[ ! -f "$preprocess_tmp_seg" ]]; then
        echo -e "${RED}Error: Segmentation output not found: $preprocess_tmp_seg${NC}" >&2
        [[ "$KEEP_TEMP" == "false" ]] && rm -rf "$temp_dir"
        exit 1
    fi

    cd "$BUNDLE_ROOT"
    python brain_t1_preprocess/revert_preprocess.py \
        "$preprocess_tmp" \
        --out "${temp_dir}/${file_basename}_revert.nii.gz" \
        --mask "$preprocess_tmp_seg" \
        --mask-out "$final_output" \
        --meta "$preprocess_meta" || {
        echo -e "${RED}Error: Reversion failed${NC}" >&2
        [[ "$KEEP_TEMP" == "false" ]] && rm -rf "$temp_dir"
        exit 1
    }

    # Clean up temporary files if not keeping them
    if [[ "$KEEP_TEMP" == "false" ]]; then
        echo -e "${YELLOW}Cleaning up temporary files...${NC}"
        rm -rf "$temp_dir"
        # Also clean up the preprocessed output directory if it only contains temp files
        local preprocess_output_dir="${output_dir}/${file_basename}_preprocessed"
        if [[ -d "$preprocess_output_dir" ]]; then
            rm -rf "$preprocess_output_dir"
        fi
    else
        echo -e "${GREEN}Temporary files kept in: $temp_dir${NC}"
    fi

    echo -e "${GREEN}✓ Successfully processed: $input_file${NC}"
    echo -e "${GREEN}  Output saved to: $final_output${NC}"
}

# Function to process a folder (batch mode)
process_folder() {
    local input_folder="$1"
    local output_dir="${OUTPUT_DIR:-./eval}"

    if [[ ! -d "$input_folder" ]]; then
        echo -e "${RED}Error: Input folder not found: $input_folder${NC}" >&2
        exit 1
    fi

    # Get absolute paths
    input_folder=$(realpath "$input_folder")
    output_dir=$(realpath -m "$output_dir")
    mkdir -p "$output_dir"

    # Find all NIfTI files
    local files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(find "$input_folder" -maxdepth 1 -type f \( -name "*.nii.gz" -o -name "*.nii" \) -print0)

    if [[ ${#files[@]} -eq 0 ]]; then
        echo -e "${YELLOW}Warning: No NIfTI files found in $input_folder${NC}" >&2
        exit 1
    fi

    echo -e "${GREEN}Found ${#files[@]} file(s) to process${NC}"

    # Process each file
    local success_count=0
    local fail_count=0

    for file in "${files[@]}"; do
        echo ""
        echo -e "${GREEN}========================================${NC}"
        if process_single_file "$file"; then
            ((success_count++))
        else
            ((fail_count++))
            echo -e "${RED}Failed to process: $file${NC}" >&2
        fi
    done

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Batch processing complete!${NC}"
    echo -e "${GREEN}  Successful: $success_count${NC}"
    if [[ $fail_count -gt 0 ]]; then
        echo -e "${RED}  Failed: $fail_count${NC}"
    fi
}

# Parse command line arguments
INPUT_FILE=""
INPUT_FOLDER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --input_folder)
            INPUT_FOLDER="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --keep-temp)
            KEEP_TEMP=true
            shift
            ;;
        --modality)
            MODALITY="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            usage
            ;;
    esac
done

# Validate arguments
if [[ -z "$INPUT_FILE" && -z "$INPUT_FOLDER" ]]; then
    echo -e "${RED}Error: Either --input or --input_folder must be specified${NC}" >&2
    usage
fi

if [[ -n "$INPUT_FILE" && -n "$INPUT_FOLDER" ]]; then
    echo -e "${RED}Error: Cannot specify both --input and --input_folder${NC}" >&2
    usage
fi

# Validate modality
if [[ ! "$MODALITY" =~ ^(MRI_BRAIN|MRI_BODY|CT_BODY)$ ]]; then
    echo -e "${YELLOW}Warning: Unknown modality '$MODALITY'. Using MRI_BRAIN.${NC}" >&2
    MODALITY="MRI_BRAIN"
fi

# Check and activate conda environment
check_conda_env

# Change to bundle root directory
cd "$BUNDLE_ROOT"

# Process based on input type
if [[ -n "$INPUT_FILE" ]]; then
    process_single_file "$INPUT_FILE"
else
    process_folder "$INPUT_FOLDER"
fi

echo -e "${GREEN}All done!${NC}"
