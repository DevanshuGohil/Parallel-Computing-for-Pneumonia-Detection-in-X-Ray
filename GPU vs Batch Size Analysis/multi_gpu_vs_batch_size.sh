#!/bin/bash
echo "Starting script..."

# Define the values for num_gpus and batch_size
num_gpus=(1 2 4)
batch_sizes=(32 64 128 256)

echo "Starting loop..."

# Loop over all combinations of num_gpus and batch_size
for ngpu in "${num_gpus[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        # Define the output file name based on the combination of parameters
        output_file="output_ngpu_${ngpu}_batch_${batch_size}.txt"
        
        # Run the Python script with the current combination of parameters
        echo "Running Python script with num_gpus=$ngpu and batch_size=$batch_size, saving output to $output_file..."
        python Pneumonia_ddp.py --num_gpus $ngpu --batch_size $batch_size > $output_file
    done
done

echo "Script finished."
