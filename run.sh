#!/bin/bash

# Define a default value for the flag
comment=false  # Flag will be false by default

# Process options and arguments using getopts
while getopts ":c" opt; do
  case $opt in
    c) comment=true ;;  # Set comment flag to true if "-c" is present
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

# Shift arguments to remove processed options
shift $((OPTIND-1))

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 [-c] <algorithm> <filename> <max_sec>"
    exit 1
fi

algo=$1
filename=$2
threshold=$3

if [[ $comment == true ]]; then
   cargo run --release --bin "$algo" "$filename" "$threshold" -c
else
  cargo run --release --bin "$algo" "$filename" "$threshold"
fi
