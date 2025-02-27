#!/bin/bash

# Check if the required argument (image name) is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <image_name> [source_folder] [dataset_folder] [script_to_run] [script_args...]"
  exit 1
fi

ROOT_NAME="workspace"
# Get the image name from arguments
IMAGE_NAME="$1"

# Optional parameters for script, source folder, and dataset folder
SRC_FOLDER="${2:-}"
DATASET_FOLDER="${3:-}"
SCRIPT_NAME="${4:-}"

# Function to extract the last directory name from a given path
get_last_dir_name() {
    echo "$(basename "$1")"
}

# Variable to track eager mode
EAGER_MODE=false

# Parse arguments for --eager flag
for arg in "$@"; do
  if [ "$arg" == "--eager" ]; then
    EAGER_MODE=true
    # Remove --eager from the arguments
    set -- "${@/--eager/}"
    # Clean up empty arguments
    args=()
    for i in "$@"; do
      [ -n "$i" ] && args+=("$i")
    done
    set -- "${args[@]}"
    break
  fi
done

# If a source folder is provided, mount it
if [ -n "$SRC_FOLDER" ]; then
  SRC_DIR_NAME=$(get_last_dir_name "$SRC_FOLDER")
  SRC_MOUNT="-v $SRC_FOLDER:/$ROOT_NAME/$SRC_DIR_NAME"
else
  SRC_MOUNT=""
fi

# If a dataset folder is provided, mount it
if [ -n "$DATASET_FOLDER" ]; then
  DATASET_DIR_NAME=$(get_last_dir_name "$DATASET_FOLDER")
  DATASET_MOUNT="-v $DATASET_FOLDER:/$ROOT_NAME/$DATASET_DIR_NAME"
else
  DATASET_MOUNT=""
fi

# Set environment variables for Docker
ENV_VARS="-e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN"
# Add eager mode environment variable if enabled
if [ "$EAGER_MODE" = true ]; then
  ENV_VARS="$ENV_VARS -e PT_HPU_LAZY_MODE=0"
  echo "Running in eager mode (PT_HPU_LAZY_MODE=0)"
fi

# Check if the image name is "base" and run vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest
if [ "$IMAGE_NAME" == "base" ]; then
  IMAGE_NAME="vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest"
fi

# Check if a script is provided, if not, run bash interactively
if [ -z "$SCRIPT_NAME" ]; then
  echo "No script provided. Starting the Docker container in interactive bash mode..."
  sudo docker run -it --runtime=habana \
  $ENV_VARS \
  --cap-add=sys_nice --net=host --ipc=host \
  --entrypoint /bin/bash \
  $SRC_MOUNT $DATASET_MOUNT \
  $IMAGE_NAME
else
  # Collect all arguments after the 4th one (script name) as script arguments
  SCRIPT_ARGS=""
  if [ $# -gt 4 ]; then
    # Shift removes the first 4 arguments, leaving only script arguments
    shift 4
    # Pass all remaining arguments to the script
    SCRIPT_ARGS="$@"
  fi

  # Check if the script is a Python file
  if [[ "$SCRIPT_NAME" == *.py ]]; then
    echo "Running Python script '$SCRIPT_NAME' with arguments: $SCRIPT_ARGS"
    sudo docker run -it --runtime=habana \
    $ENV_VARS \
    --cap-add=sys_nice --net=host --ipc=host \
    $SRC_MOUNT $DATASET_MOUNT \
    $IMAGE_NAME \
    /$ROOT_NAME/$SRC_DIR_NAME/$SCRIPT_NAME $SCRIPT_ARGS
  else
    # Run the Docker container with the specified script
    echo "Running the script '$SCRIPT_NAME' with arguments: $SCRIPT_ARGS"
    sudo docker run -it --runtime=habana \
    $ENV_VARS \
    --cap-add=sys_nice --net=host --ipc=host \
    $SRC_MOUNT $DATASET_MOUNT \
    $IMAGE_NAME \
    /$ROOT_NAME/$SRC_DIR_NAME/$SCRIPT_NAME $SCRIPT_ARGS
  fi
fi
