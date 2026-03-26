#!/bin/bash

# Brain MRI Segmentation Script
# Supports single file, folder batch processing, or file list processing
# Supports optional skull stripping and temporary file retention

# Print startup message
echo "Starting brain segmentation script..." >&2

# Default values
KEEP_TEMP=false
OUTPUT_DIR=""
MODALITY="MRI_BRAIN"
SKIP_SKULLSTRIP=false
SKIP_EXISTING=true
NUM_PARTITIONS=1
PARTITION_NUM=1

# Get script directory - handle both direct execution and sourced execution
if [[ "${BASH_SOURCE[0]}" != "" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || {
        echo "Error: Failed to determine script directory" >&2
        exit 1
    }
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" || {
        echo "Error: Failed to determine script directory" >&2
        exit 1
    }
fi
BUNDLE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)" || {
    echo "Error: Failed to determine bundle root directory" >&2
    exit 1
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Brain MRI Segmentation Script for NV-Segment-CTMR
Supports single file, folder batch processing, or file list processing

OPTIONS:
    --input FILE              Single NIfTI file to segment
    --input_folder FOLDER     Folder containing NIfTI files to process (batch mode)
    --file_list FILE          Text file with one file path per line (relative to root_path)
    --root_path PATH          Root path where file paths in file_list are relative to
    --output_dir DIR          Output directory (default: ./eval)
    --keep-temp               Keep temporary preprocessing files (default: false)
    --no-skullstrip           Skip skull stripping step (default: false, performs skull stripping)
    --modality MODALITY       Segmentation modality: MRI_BRAIN (default), MRI_BODY, CT_BODY
    --no-skip                 Don't skip existing output files (default: skip if exists, only for file_list mode)
    --num_partitions N        Split file list into N partitions (default: 1, no split, only for file_list mode)
    --partition M             Process partition M (1-indexed, default: 1). Requires --num_partitions (only for file_list mode)
    -h, --help                Show this help message

EXAMPLES:
    # Single file processing
    $0 --input example/brain_t1.nii.gz

    # Single file without skull stripping
    $0 --input example/brain_t1.nii.gz --no-skullstrip

    # Batch processing from folder
    $0 --input_folder example/ --output_dir results/

    # Process files from a list
    $0 --file_list file_list.txt --root_path /path/to/root --output_dir /path/to/output

    # Process files from a list without skull stripping
    $0 --file_list file_list.txt --root_path /path/to/root --output_dir /path/to/output --no-skullstrip

    # Split into 10 partitions and process partition 3
    $0 --file_list file_list.txt --root_path /path/to/root --output_dir /path/to/output --num_partitions 10 --partition 3

NOTES:
    - When using --file_list, output files maintain the same directory structure as input, with _seg suffix added
    - Existing output files are skipped by default to support job resubmission (file_list mode only)

EOF
    exit 1
}

# Function to process a single file
process_single_file() {
    local input_file="$1"
    local output_file="$2"
    local failed_log="$3"
    local output_dir=$(dirname "$output_file")
    
    # Ensure output directory exists
    mkdir -p "$output_dir"
    
    # Get absolute paths
    if command -v realpath &> /dev/null; then
        input_file=$(realpath "$input_file" 2>/dev/null || echo "$input_file")
    else
        if [[ ! "$input_file" =~ ^/ ]]; then
            input_file="$(cd "$(dirname "$input_file")" && pwd)/$(basename "$input_file")"
        fi
    fi
    
    # Extract filename without extension
    local file_basename=$(basename "$input_file" .nii.gz)
    file_basename=$(basename "$file_basename" .nii)
    
    # Create temporary directory in the output directory (use a unique name to avoid conflicts)
    local temp_dir="${output_dir}/${file_basename}_temp_$$"
    mkdir -p "$temp_dir"
    
    # Temporary file paths
    local skull_stripped="${temp_dir}/${file_basename}_skull_stripped.nii.gz"
    local preprocess_tmp="${temp_dir}/${file_basename}_preprocessed.nii.gz"
    local preprocess_meta="${temp_dir}/${file_basename}_preprocessed.meta.json"
    # Monai bundle saves to output_dir from config, which defaults to bundle_root/eval
    # We'll override it to use our output_dir, and it will save as:
    # {output_dir}/{basename}_preprocessed/{basename}_preprocessed_trans.nii.gz
    local preprocess_tmp_seg="${output_dir}/${file_basename}_preprocessed/${file_basename}_preprocessed_trans.nii.gz"
    
    # Cleanup function
    cleanup_on_error() {
        if [[ "$KEEP_TEMP" == "false" ]]; then
            rm -rf "$temp_dir"
        fi
    }
    trap cleanup_on_error ERR
    
    echo -e "${GREEN}Processing: $input_file${NC}" >&2
    echo -e "${GREEN}Output will be saved to: $output_file${NC}" >&2
    
    # Function to log failure and return
    log_failure() {
        local reason="$1"
        echo -e "${RED}$reason: $input_file${NC}" >&2
        if [[ -n "$failed_log" ]]; then
            echo "$input_file" >> "$failed_log"
        fi
        cleanup_on_error
        return 1
    }
    
    # Determine which file to use for preprocessing
    local preprocess_input="$input_file"
    local step_num=1
    local total_steps=3
    
    # Step 1: Skull stripping with SynthStrip (if not skipped)
    if [[ "$SKIP_SKULLSTRIP" == "false" ]]; then
        total_steps=4
        echo -e "${YELLOW}Step 1/4: Skull stripping...${NC}" >&2
        if [[ ! -f "$skull_stripped" ]]; then
            cd "$BUNDLE_ROOT"
            ./brain_t1_preprocess/synthstrip-docker -i "$input_file" -o "$skull_stripped" || {
                log_failure "Error: Skull stripping failed"
                return 1
            }
        else
            echo -e "${YELLOW}  Skull-stripped file already exists, skipping...${NC}" >&2
        fi
        preprocess_input="$skull_stripped"
        step_num=2
    else
        echo -e "${YELLOW}Note: Skull stripping step is skipped${NC}" >&2
    fi
    
    # Step 2: Affine align to the LUMIR template
    echo -e "${YELLOW}Step ${step_num}/${total_steps}: Affine alignment to LUMIR template...${NC}" >&2
    cd "$BUNDLE_ROOT"
    python brain_t1_preprocess/preprocess.py \
        "$preprocess_input" \
        brain_t1_preprocess/LUMIR_template.nii.gz \
        "$preprocess_tmp" \
        --save-preprocess "$preprocess_meta" || {
        log_failure "Error: Preprocessing failed"
        return 1
    }
    
    # Step 3: Segment the brain
    ((step_num++))
    echo -e "${YELLOW}Step ${step_num}/${total_steps}: Running segmentation...${NC}" >&2
    cd "$BUNDLE_ROOT"
    # Override output_dir in config to use our output_dir so segmentation saves to the right place
    python -m monai.bundle run \
        --config_file configs/inference.json \
        --input_dict "{'image':'$preprocess_tmp'}" \
        --output_dir "$output_dir" \
        --modality "$MODALITY" || {
        log_failure "Error: Segmentation failed"
        return 1
    }
    
    # Step 4: Revert the segmentation back to original space
    ((step_num++))
    echo -e "${YELLOW}Step ${step_num}/${total_steps}: Reverting to original space...${NC}" >&2
    if [[ ! -f "$preprocess_tmp_seg" ]]; then
        log_failure "Error: Segmentation output not found"
        return 1
    fi
    
    cd "$BUNDLE_ROOT"
    python brain_t1_preprocess/revert_preprocess.py \
        "$preprocess_tmp" \
        --out "${temp_dir}/${file_basename}_revert.nii.gz" \
        --mask "$preprocess_tmp_seg" \
        --mask-out "$output_file" \
        --meta "$preprocess_meta" || {
        log_failure "Error: Reversion failed"
        return 1
    }
    
    # Clean up temporary files if not keeping them
    if [[ "$KEEP_TEMP" == "false" ]]; then
        echo -e "${YELLOW}Cleaning up temporary files...${NC}" >&2
        rm -rf "$temp_dir"
        # Also clean up the preprocessed output directory if it only contains temp files
        local preprocess_output_dir="${output_dir}/${file_basename}_preprocessed"
        if [[ -d "$preprocess_output_dir" ]]; then
            rm -rf "$preprocess_output_dir"
        fi
    else
        echo -e "${GREEN}Temporary files kept in: $temp_dir${NC}" >&2
    fi
    
    trap - ERR
    
    echo -e "${GREEN}✓ Successfully processed: $input_file${NC}" >&2
    echo -e "${GREEN}  Output saved to: $output_file${NC}" >&2
    return 0
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
    if command -v realpath &> /dev/null; then
        input_folder=$(realpath "$input_folder" 2>/dev/null || echo "$input_folder")
        output_dir=$(realpath -m "$output_dir" 2>/dev/null || echo "$output_dir")
    else
        if [[ ! "$input_folder" =~ ^/ ]]; then
            input_folder="$(cd "$input_folder" && pwd)"
        fi
        if [[ ! "$output_dir" =~ ^/ ]]; then
            output_dir="$(cd "$(dirname "$output_dir")" && pwd)/$(basename "$output_dir")"
        fi
    fi
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

    echo -e "${GREEN}Found ${#files[@]} file(s) to process${NC}" >&2

    # Process each file
    local success_count=0
    local fail_count=0

    for file in "${files[@]}"; do
        echo "" >&2
        echo -e "${GREEN}========================================${NC}" >&2
        
        # Determine output file path
        local file_basename=$(basename "$file" .nii.gz)
        file_basename=$(basename "$file_basename" .nii)
        local rel_ext=""
        if [[ "$file" == *.nii.gz ]]; then
            rel_ext=".nii.gz"
        elif [[ "$file" == *.nii ]]; then
            rel_ext=".nii"
        fi
        local output_file="${output_dir}/${file_basename}_trans${rel_ext}"
        
        if process_single_file "$file" "$output_file" ""; then
            ((success_count++))
        else
            ((fail_count++))
            echo -e "${RED}Failed to process: $file${NC}" >&2
        fi
    done

    echo "" >&2
    echo -e "${GREEN}========================================${NC}" >&2
    echo -e "${GREEN}Batch processing complete!${NC}" >&2
    echo -e "${GREEN}  Successful: $success_count${NC}" >&2
    if [[ $fail_count -gt 0 ]]; then
        echo -e "${RED}  Failed: $fail_count${NC}" >&2
    fi
}

# Function to process files from txt file list
process_file_list() {
    local file_list="$1"
    local root_path="$2"
    local output_dir="${OUTPUT_DIR:-./eval}"
    
    if [[ ! -f "$file_list" ]]; then
        echo -e "${RED}Error: File list not found: $file_list${NC}" >&2
        exit 1
    fi
    
    if [[ ! -d "$root_path" ]]; then
        echo -e "${RED}Error: Root path not found: $root_path${NC}" >&2
        exit 1
    fi
    
    # Get absolute paths
    if command -v realpath &> /dev/null; then
        file_list=$(realpath "$file_list" 2>/dev/null || echo "$file_list")
        root_path=$(realpath "$root_path" 2>/dev/null || echo "$root_path")
        output_dir=$(realpath -m "$output_dir" 2>/dev/null || echo "$output_dir")
    else
        # Fallback if realpath is not available
        if [[ ! "$file_list" =~ ^/ ]]; then
            file_list="$(cd "$(dirname "$file_list")" && pwd)/$(basename "$file_list")"
        fi
        if [[ ! "$root_path" =~ ^/ ]]; then
            root_path="$(cd "$root_path" && pwd)"
        fi
        if [[ ! "$output_dir" =~ ^/ ]]; then
            output_dir="$(cd "$(dirname "$output_dir")" && pwd)/$(basename "$output_dir")"
        fi
    fi
    mkdir -p "$output_dir"
    
    # Create log file for failed/timeout files (after directory is created)
    local failed_log="${output_dir}/failed_files_$(date +%Y%m%d_%H%M%S).txt"
    touch "$failed_log"
    echo -e "${YELLOW}Failed/timeout files will be logged to: $failed_log${NC}" >&2
    
    # Read file paths from the list
    local files=()
    local line_num=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_num++))
        # Skip empty lines and comments
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        if [[ -z "$line" ]] || [[ "$line" =~ ^# ]]; then
            continue
        fi
        
        # Remove leading ./ if present
        line="${line#./}"
        
        # Construct full path
        local full_path="${root_path}/${line}"
        
        if [[ ! -f "$full_path" ]]; then
            echo -e "${YELLOW}Warning: File not found (line $line_num): $full_path${NC}" >&2
            continue
        fi
        
        files+=("$full_path")
    done < "$file_list"
    
    if [[ ${#files[@]} -eq 0 ]]; then
        echo -e "${YELLOW}Warning: No valid files found in $file_list${NC}" >&2
        exit 1
    fi
    
    # Sort files deterministically for consistent partitioning
    local sorted_files=()
    while IFS= read -r line; do
        sorted_files+=("$line")
    done < <(printf '%s\n' "${files[@]}" | sort)
    files=("${sorted_files[@]}")
    
    # Apply partitioning if requested
    local total_files=${#files[@]}
    local partition_files=()
    
    if [[ $NUM_PARTITIONS -gt 1 ]]; then
        if [[ $PARTITION_NUM -lt 1 ]] || [[ $PARTITION_NUM -gt $NUM_PARTITIONS ]]; then
            echo -e "${RED}Error: Partition number must be between 1 and $NUM_PARTITIONS${NC}" >&2
            exit 1
        fi
        
        # Calculate partition boundaries (deterministic split)
        local files_per_partition=$((total_files / NUM_PARTITIONS))
        local remainder=$((total_files % NUM_PARTITIONS))
        
        # Calculate start and end indices for this partition (0-indexed)
        local start_idx=0
        for ((i=1; i<PARTITION_NUM; i++)); do
            local part_size=$files_per_partition
            if [[ $i -le $remainder ]]; then
                ((part_size++))
            fi
            ((start_idx += part_size))
        done
        
        local end_idx=$start_idx
        local part_size=$files_per_partition
        if [[ $PARTITION_NUM -le $remainder ]]; then
            ((part_size++))
        fi
        ((end_idx += part_size))
        
        # Extract partition files
        for ((i=start_idx; i<end_idx && i<total_files; i++)); do
            partition_files+=("${files[i]}")
        done
        
        echo -e "${GREEN}Total files in list: $total_files${NC}" >&2
        echo -e "${GREEN}Partition $PARTITION_NUM of $NUM_PARTITIONS: ${#partition_files[@]} file(s)${NC}" >&2
        files=("${partition_files[@]}")
    else
        echo -e "${GREEN}Found ${#files[@]} file(s) to process${NC}" >&2
    fi
    
    echo -e "${BLUE}Root path: $root_path${NC}" >&2
    echo -e "${BLUE}Output directory: $output_dir${NC}" >&2
    
    # Process each file
    local success_count=0
    local fail_count=0
    local skip_count=0
    local total_in_partition=${#files[@]}
    local processed_count=0
    
    for input_file in "${files[@]}"; do
        ((processed_count++))
        local remaining=$((total_in_partition - processed_count))
        
        echo "" >&2
        echo -e "${GREEN}========================================${NC}" >&2
        echo -e "${BLUE}Progress: [$((processed_count-1))/$total_in_partition] completed, $((remaining+1)) remaining${NC}" >&2
        
        # Get relative path from root
        local rel_path="${input_file#$root_path/}"
        
        # Construct output path maintaining directory structure
        # Change filename to add _seg before extension
        local rel_dir=$(dirname "$rel_path")
        local rel_filename=$(basename "$rel_path")
        local rel_basename=$(basename "$rel_filename" .nii.gz)
        rel_basename=$(basename "$rel_basename" .nii)
        local rel_ext=""
        if [[ "$rel_filename" == *.nii.gz ]]; then
            rel_ext=".nii.gz"
        elif [[ "$rel_filename" == *.nii ]]; then
            rel_ext=".nii"
        fi
        
        local output_file="${output_dir}/${rel_dir}/${rel_basename}_seg${rel_ext}"
        
        # Check if output already exists (before processing)
        if [[ "$SKIP_EXISTING" == "true" ]] && [[ -f "$output_file" ]]; then
            echo -e "${BLUE}Skipping (output exists): $input_file${NC}" >&2
            echo -e "${BLUE}  Output: $output_file${NC}" >&2
            ((skip_count++))
            continue
        fi
        
        # Process the file with overall timeout of 5 minutes
        local timeout_seconds=300  # 5 minutes total per scan
        local process_result=0
        
        if command -v timeout &> /dev/null; then
            # Use timeout command to limit total processing time per scan
            # Export necessary variables for the function
            export BUNDLE_ROOT KEEP_TEMP MODALITY SKIP_SKULLSTRIP
            
            # Export the function so it's available in subshell
            # If export -f fails, we'll declare it inline in bash -c
            export -f process_single_file 2>/dev/null
            
            # Run with timeout - use bash -c to ensure function is available
            # Escape file paths safely using printf %q (bash-recommended method)
            printf -v escaped_input_file %q "$input_file"
            printf -v escaped_output_file %q "$output_file"
            printf -v escaped_failed_log %q "$failed_log"
            
            timeout $timeout_seconds bash -c "
                $(declare -f process_single_file)
                process_single_file $escaped_input_file $escaped_output_file $escaped_failed_log
            " 2>&1
            local exit_code=$?
            
            if [[ $exit_code -eq 124 ]]; then
                # Timeout occurred (exit code 124 is timeout)
                echo -e "${RED}Error: Processing timed out after ${timeout_seconds}s: $input_file${NC}" >&2
                echo "$input_file" >> "$failed_log"
                ((fail_count++))
                process_result=1
            elif [[ $exit_code -eq 0 ]]; then
                # Process completed, check if output was created
                if [[ -f "$output_file" ]]; then
                    ((success_count++))
                    process_result=0
                else
                    # Output not created despite success exit code
                    echo -e "${RED}Error: Output file not created: $input_file${NC}" >&2
                    echo "$input_file" >> "$failed_log"
                    ((fail_count++))
                    process_result=1
                fi
            else
                # Process failed (error already logged in process_single_file)
                ((fail_count++))
                process_result=1
            fi
        else
            # Fallback: run without timeout wrapper
            # Note: Without timeout command, we can't enforce the 5-minute limit
            # but we can still catch failures
            if process_single_file "$input_file" "$output_file" "$failed_log"; then
                ((success_count++))
                process_result=0
            else
                ((fail_count++))
                process_result=1
            fi
        fi
    done
    
    echo "" >&2
    echo -e "${GREEN}========================================${NC}" >&2
    echo -e "${GREEN}Batch processing complete!${NC}" >&2
    echo -e "${GREEN}  Successful: $success_count${NC}" >&2
    if [[ $skip_count -gt 0 ]]; then
        echo -e "${BLUE}  Skipped (existing): $skip_count${NC}" >&2
    fi
    if [[ $fail_count -gt 0 ]]; then
        echo -e "${RED}  Failed/Timeout: $fail_count${NC}" >&2
        echo -e "${YELLOW}  Failed files logged to: $failed_log${NC}" >&2
    fi
}

# Parse command line arguments
INPUT_FILE=""
INPUT_FOLDER=""
FILE_LIST=""
ROOT_PATH=""

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
        --file_list)
            FILE_LIST="$2"
            shift 2
            ;;
        --root_path)
            ROOT_PATH="$2"
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
        --no-skullstrip)
            SKIP_SKULLSTRIP=true
            shift
            ;;
        --modality)
            MODALITY="$2"
            shift 2
            ;;
        --no-skip)
            SKIP_EXISTING=false
            shift
            ;;
        --num_partitions)
            NUM_PARTITIONS="$2"
            shift 2
            ;;
        --partition)
            PARTITION_NUM="$2"
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
input_count=0
[[ -n "$INPUT_FILE" ]] && ((input_count++))
[[ -n "$INPUT_FOLDER" ]] && ((input_count++))
[[ -n "$FILE_LIST" ]] && ((input_count++))

if [[ $input_count -eq 0 ]]; then
    echo -e "${RED}Error: One of --input, --input_folder, or --file_list must be specified${NC}" >&2
    usage
fi

if [[ $input_count -gt 1 ]]; then
    echo -e "${RED}Error: Cannot specify multiple input options (--input, --input_folder, --file_list)${NC}" >&2
    usage
fi

if [[ -n "$FILE_LIST" ]] && [[ -z "$ROOT_PATH" ]]; then
    echo -e "${RED}Error: --root_path must be specified when using --file_list${NC}" >&2
    usage
fi

# Validate partition arguments
if [[ $NUM_PARTITIONS -lt 1 ]]; then
    echo -e "${RED}Error: --num_partitions must be at least 1${NC}" >&2
    usage
fi

if [[ $PARTITION_NUM -lt 1 ]] || [[ $PARTITION_NUM -gt $NUM_PARTITIONS ]]; then
    echo -e "${RED}Error: --partition must be between 1 and $NUM_PARTITIONS${NC}" >&2
    usage
fi

# Validate modality
if [[ ! "$MODALITY" =~ ^(MRI_BRAIN|MRI_BODY|CT_BODY)$ ]]; then
    echo -e "${YELLOW}Warning: Unknown modality '$MODALITY'. Using MRI_BRAIN.${NC}" >&2
    MODALITY="MRI_BRAIN"
fi

# Change to bundle root directory
if [[ ! -d "$BUNDLE_ROOT" ]]; then
    echo -e "${RED}Error: Bundle root directory not found: $BUNDLE_ROOT${NC}" >&2
    exit 1
fi

cd "$BUNDLE_ROOT" || {
    echo -e "${RED}Error: Failed to change to bundle root directory: $BUNDLE_ROOT${NC}" >&2
    exit 1
}

# Process based on input type
if [[ -n "$INPUT_FILE" ]]; then
    # Single file mode
    file_basename=$(basename "$INPUT_FILE" .nii.gz)
    file_basename=$(basename "$file_basename" .nii)
    rel_ext=""
    if [[ "$INPUT_FILE" == *.nii.gz ]]; then
        rel_ext=".nii.gz"
    elif [[ "$INPUT_FILE" == *.nii ]]; then
        rel_ext=".nii"
    fi
    output_file="${OUTPUT_DIR:-./eval}/${file_basename}_trans${rel_ext}"
    process_single_file "$INPUT_FILE" "$output_file" ""
elif [[ -n "$INPUT_FOLDER" ]]; then
    # Folder batch mode
    process_folder "$INPUT_FOLDER"
else
    # File list mode
    process_file_list "$FILE_LIST" "$ROOT_PATH"
fi

echo -e "${GREEN}All done!${NC}" >&2
