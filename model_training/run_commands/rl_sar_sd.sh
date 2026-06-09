set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_MODEL_PATH="YOUR_BASE_MODEL_PATH"  # Set to your base model path (e.g. /path/to/qwen3-8b)

export CUDA_VISIBLE_DEVICES=0,1,2,3

unset WANDB_DIR  # 清除可能的目录配置
export WANDB_API_KEY="YOUR_WANDB_API_KEY"  # Get your API key from: https://wandb.ai/authorize
export WANDB_PROJECT="PyGeoX-RL-QWEN-8b-DENSE"
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
# Re-create symlink each run so it always points to the correct repo location
rm -f $SITE_PACKAGES/pygeox
ln -sf $PROJECT_ROOT/pygeox $SITE_PACKAGES/pygeox

# --- DEEP REPAIR ENV VARS ---
#export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export VLLM_CONFIGURE_LOGGING=0
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_TIMEOUT=3600
export DEEPSPEED_TIMEOUT=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# -----------------------------

# --- RAY STABILITY ENV VARS ---
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_memory_monitor_refresh_ms=0
export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/tmp/ray_spill_ours"}}'
# -----------------------------

# --- OPENRLHF ASYNC TASKS FIX ---
# Default is 128, but this causes system resource exhaustion and worker crashes
# Lower it to prevent "Check failed: objects_valid" errors
export OPENRLHF_ASYNC_NUM_TASKS=4
# -----------------------------

#next try remove all previous export, point to Xuejun folder and go back in ray command
#use Xuejun folder
#export PYTHONPATH=/mnt/disk0/y84370147/Prover_training/kimina-rl/prover-openrlhf/ProverTraining:$PYTHONPATH


# === PARALLEL-SAFE Ray Cleanup ===
# Only kill processes on THIS cluster's port (6381), not all Ray processes
# This allows multiple training scripts to run in parallel on different GPUs
MY_RAY_PORT=6381
MY_DASHBOARD_PORT=8266

echo "=== Cleaning up Ray cluster on port $MY_RAY_PORT only ==="
# Kill any process using our specific ports
lsof -ti:$MY_RAY_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$MY_DASHBOARD_PORT | xargs kill -9 2>/dev/null || true
sleep 2

# Start Ray cluster with reasonable object store memory (50GB for 4-GPU runs)
unset RAY_ADDRESS
rm -rf /tmp/ray_spill_ours; mkdir -p /tmp/ray_spill_ours
ray start --head --port=$MY_RAY_PORT --num-gpus=4 --object-store-memory=50000000000 --dashboard-port=$MY_DASHBOARD_PORT

# CRITICAL: Set RAY_ADDRESS so openrlhf connects to THIS cluster, not another one
export RAY_ADDRESS="127.0.0.1:$MY_RAY_PORT"
sleep 2

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

#kill 3443047
#ray stop --force
#shuf $PROJECT_ROOT/model_training/rl_data_new_medium.jsonl > $PROJECT_ROOT/model_training/rl_data_new_medium_shuf.jsonl
python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.4 \
   --gradient_checkpointing \
   --adam_offload \
   --init_kl_coef 0.01 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --eps_clip_low_high 0.2 0.3 \
   --pretrain $BASE_MODEL_PATH \
   --remote_rm_url $PROJECT_ROOT/model_training/run_commands/reward_func_sar_sd.py \
   --save_path $PROJECT_ROOT/model_training/RL_checkpoint_8b_dense \
   --ckpt_path $PROJECT_ROOT/model_training/RL_checkpoint_8b_dense \
   --load_checkpoint \
   --save_steps 3 \
   --max_ckpt_num 25 \
   --micro_train_batch_size 2 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 32 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 4096 \
   --max_samples 100000 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --attn_implementation flash_attention_2 \
   --actor_learning_rate 1e-6 \
   --prompt_data $PROJECT_ROOT/model_training/rl_data_new_medium_shuf.jsonl \
   --input_key messages \
   --label_key source_file \
   --apply_chat_template \
   --vllm_sync_backend nccl \
   --logging_steps 1 \
   --use_wandb $WANDB_API_KEY \
   --wandb_project "PyGeoX-RL-LLM-DENSE-REWARD-8b"

#--vllm_enable_sleep
#--deepspeed_enable_sleep
#--save_hf_ckpt
#--packing_samples \

#later consider:
#--lr_scheduler_type constant_with_warmup \
#--actor_learning_rate 5e-6 \  # Keep this the same
#--warmup_ratio 0.03 \         # 3% warmup (standard)
