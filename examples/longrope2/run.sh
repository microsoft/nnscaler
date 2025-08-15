SCRIPT_DIR=$(dirname "$(realpath "$0")")
EXAMPLE_DIR=$(dirname "$SCRIPT_DIR")  
MAGICCUBE_DIR=$(dirname "$EXAMPLE_DIR")
export PYTHONPATH=$PYTHONPATH:$MAGICCUBE_DIR:$EXAMPLE_DIR

# download data to at MagicCube/examples/longrope2/data, will take around 100GB disk memory.
python data/download.py
# process the data to mix context window length format for long context training, will take around 900GB disk memory.
python data/process.py --tokenizer_name_or_path "meta-llama/Meta-Llama-3-8B"
# compile the distributed code for llama3 model with dp2, tp4 on 8 gpus
python train.py --run_mode compile --model_id "meta-llama/Meta-Llama-3-8B" --model_config llama3_8b_longrope2_config.json --dataset_path data/mix-context-win-short-8192-long-131072 --plan_ngpus=4 --runtime_ngpus=8 --recompute_modules LlamaDecoderLayer --gpu_mem_constraint 64 --enable-chunk-loss --grad_accumulation_steps 16 --max_train_steps 2250 2>&1 | tee compile.log
# run the training job
torchrun --nproc_per_node=8 train.py --model_id "meta-llama/Meta-Llama-3-8B" --model_config llama3_8b_longrope2_config.json --dataset_path data/mix-context-win-short-8192-long-131072 --plan_ngpus=4 --runtime_ngpus=8 --recompute_modules LlamaDecoderLayer --gpu_mem_constraint 64 --enable-chunk-loss --grad_accumulation_steps 16 --max_train_steps 2250 2>&1 | tee run.log
