#!/bin/bash
# set -e

# COMMON
HIDDEN=4096
HIDDEN_TINY=128
GBS=512
GBS_TINY=32
SEQLEN=128
seeds=(0)

# 70B setup
LAYERS=80
HEADS=64

# 8B setup
LAYERS=32
HEADS=32


# small setup
# LAYERS=2
# HEADS=32

# selected experiments
# nm_sizes=()
# tp_sizes=()
# dp_sizes=()
# layers=()
# gb_sizes=()
# pp_sizes=()
# hidden_sizes=()
# seq_sizes=(256)
# head_sizes=()

# # full experiments
# nm_sizes=(1 2 4 8 16 32)
nm_sizes=(2 4 8 16 32)
tp_sizes=(1 2 4 8)
dp_sizes=(2 8 16 32 64)
layers=(4 8 16 32)
pp_sizes=(2 4 8 16)
gb_sizes=(512 1024 2048 4096)
head_sizes=(8 16 32 64 128)
# gb_sizes=(512 1024 2048)
hidden_sizes=(1024 4096 8192 16384)
# hidden_sizes=(1024 4096 8192)
# seq_sizes=(256 1024 4096 8192 131072)
seq_sizes=(128 256 1024 4096 8192)
# seq_sizes=(256 1024 4096)

# # rerun experiments
# # nm_sizes=(1 2 4 8 16 32)
# nm_sizes=()
# tp_sizes=()
# dp_sizes=()
# layers=()
# pp_sizes=(2 4 8 16)
# # pp_sizes=()
# # gb_sizes=()
# gb_sizes=(512 1024 2048 4096)
# # head_sizes=()
# head_sizes=(8 16 32 64 128)
# # gb_sizes=(512 1024 2048)
# # hidden_sizes=()
# hidden_sizes=(1024 4096 8192 16384)
# # hidden_sizes=(1024 4096 8192)
# seq_sizes=(128 256 1024 4096 8192)
# # seq_sizes=()
# # seq_sizes=(256 1024 4096)

# # partial experiments
# nm_sizes=(1 2 4)
# tp_sizes=(1 2)
# dp_sizes=(2 4)
# layers=(2 3)
# pp_sizes=(2)
# head_sizes=(8 16)
# gb_sizes=(512)
# hidden_sizes=(1024 4096)
# seq_sizes=(256)

# # trial experiments
# nm_sizes=(1 2 4)
# tp_sizes=()
# dp_sizes=()
# layers=()
# head_sizes=()
# pp_sizes=()
# gb_sizes=()
# hidden_sizes=()
# seq_sizes=()



# cmd file
LOG_DIR="data/logs"
mkdir -p $LOG_DIR
CMD_FILE="scripts/run_trends.sh"
printf "#!/bin/bash\n" > $CMD_FILE

# cmds
CMD="python main.py --sm %s --pm %s --seed %d --no_cache_nodes --no_cache_stages --time  --max_vrf_proc 30 --max_ser_proc 30 --loglevel INFO |& tee %s"
MGFNAME="llama3_default_dp%d_pp%d_tp%d_nm%d_gbs%d_ly%d_h%d_hi%d_sq%d"
MGPATH="gen_model/mgeners/%s.pkl"
LOGPATH="$LOG_DIR/%s.txt"

dump_cmd(){
    gbs=$1
    ly=$2
    h=$3
    hi=$4
    seqlen=$5
    pm_dp=$6
    pm_pp=$7
    pm_tp=$8
    pm_nm=$9
    sm_fname=$(printf "$MGFNAME" 1 1 1 1 $gbs $ly $h $hi $seqlen)
    pm_fname=$(printf "$MGFNAME" $pm_dp $pm_pp $pm_tp $pm_nm $gbs $ly $h $hi $seqlen)
    sm_path=$(printf "$MGPATH" $sm_fname)
    pm_path=$(printf "$MGPATH" $pm_fname)
    logpath=$(printf "$LOGPATH" $pm_fname)
    for seed in "${seeds[@]}"; do
        cmd=$(printf "$CMD" "$sm_path" "$pm_path" "$seed" "$logpath")
        printf "%s\n" "$cmd" >> "$CMD_FILE"
    done
    
}

# echo "GENERATING real 8B"
# printf "# 8B\n" >> $CMD_FILE
# PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
#     --nproc_per_node=1  \
#     gen_model/gen_llama3_default.py --policy dp \
#         --layers 32 \
#         --hidden $HIDDEN \
#         --heads 32 \
#         --dp_size 512 \
#         --pp_size 1 \
#         --tp_size 1 \
#         --gbs $GBS \
#         --mbs $GBS \
#         --seqlen $SEQLEN
# dump_cmd $GBS 32 32 $HIDDEN $SEQLEN 512 1 1 1

echo "GENERATING SINGLE"
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=1  \
    gen_model/gen_llama3_default.py --policy dp \
        --layers $LAYERS \
        --hidden $HIDDEN\
        --heads $HEADS \
        --dp_size 1 \
        --pp_size 1 \
        --tp_size 1 \
        --gbs $GBS \
        --mbs $GBS \
        --seqlen $SEQLEN

echo "GENERATING NM"
printf "# NM\n" >> $CMD_FILE
for nm in "${nm_sizes[@]}"; do
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy hybrid \
            --layers $LAYERS \
            --hidden $HIDDEN \
            --heads $HEADS \
            --dp_size 16 \
            --pp_size 4 \
            --tp_size 8 \
            --gbs $GBS \
            --mbs $((GBS/16/nm)) \
            --seqlen $SEQLEN
    dump_cmd $GBS $LAYERS $HEADS $HIDDEN $SEQLEN 16 4 8 $nm
done

echo "GENERATING TP"
printf "# TP\n" >> $CMD_FILE
for tp_size in "${tp_sizes[@]}"; do
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy hybrid \
            --layers $LAYERS \
            --hidden $HIDDEN \
            --heads $HEADS \
            --dp_size 16 \
            --pp_size 4 \
            --tp_size $tp_size \
            --gbs $GBS \
            --mbs 4 \
            --seqlen $SEQLEN
    dump_cmd $GBS $LAYERS $HEADS $HIDDEN $SEQLEN 16 4 $tp_size $((GBS/16/4))
done

echo "GENERATING DP"
printf "# DP\n" >> $CMD_FILE
for dp_size in "${dp_sizes[@]}"; do
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy hybrid \
            --layers $LAYERS \
            --hidden $HIDDEN \
            --heads $HEADS \
            --dp_size $dp_size \
            --pp_size 4 \
            --tp_size 8 \
            --gbs $GBS \
            --mbs $((GBS/dp_size/8)) \
            --seqlen $SEQLEN
    dump_cmd $GBS $LAYERS $HEADS $HIDDEN $SEQLEN $dp_size 4 8 8
done

echo "GENERATING LAYERS"
printf "# LAYERS\n" >> $CMD_FILE
for layer in "${layers[@]}"; do
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy dp \
            --layers $layer \
            --hidden $HIDDEN \
            --heads $HEADS \
            --dp_size 1 \
            --pp_size 1 \
            --tp_size 1 \
            --gbs $GBS \
            --mbs $GBS \
            --seqlen $SEQLEN
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy hybrid \
            --layers $layer \
            --hidden $HIDDEN \
            --heads $HEADS \
            --dp_size 16 \
            --pp_size 4 \
            --tp_size 8 \
            --gbs $GBS \
            --mbs 4 \
            --seqlen $SEQLEN
    dump_cmd $GBS $layer $HEADS $HIDDEN $SEQLEN 16 4 8 $((GBS/16/4))
done

echo "GENERATING PP"
printf "# PP\n" >> $CMD_FILE
for pp_size in "${pp_sizes[@]}"; do
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy hybrid \
            --layers $LAYERS \
            --hidden $HIDDEN \
            --heads $HEADS \
            --dp_size 2 \
            --pp_size $pp_size \
            --tp_size 2 \
            --gbs $GBS \
            --mbs 128 \
            --seqlen $SEQLEN
    dump_cmd $GBS $LAYERS $HEADS $HIDDEN $SEQLEN 2 $pp_size 2 $((GBS/2/128))
done

seeds=(0)

echo "GENERATING GBS"
printf "# GBS\n" >> $CMD_FILE
for gbs in "${gb_sizes[@]}"; do
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy dp \
            --layers $LAYERS \
            --hidden $HIDDEN \
            --heads $HEADS \
            --dp_size 1 \
            --pp_size 1 \
            --tp_size 1 \
            --gbs $gbs \
            --mbs $gbs \
            --seqlen $SEQLEN
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy hybrid \
            --layers $LAYERS \
            --hidden $HIDDEN \
            --heads $HEADS \
            --dp_size 2 \
            --pp_size 2 \
            --tp_size 2 \
            --gbs $gbs \
            --mbs $((gbs/2/2)) \
            --seqlen $SEQLEN
    dump_cmd $gbs $LAYERS $HEADS $HIDDEN $SEQLEN 2 2 2 2
done

echo "GENERATING HEADS"
printf "# HEADS\n" >> $CMD_FILE
for heads in "${head_sizes[@]}"; do
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy dp \
            --layers $LAYERS \
            --hidden $HIDDEN \
            --heads $heads \
            --dp_size 1 \
            --pp_size 1 \
            --tp_size 1 \
            --gbs $GBS \
            --mbs $GBS \
            --seqlen $SEQLEN
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy hybrid \
            --layers $LAYERS \
            --hidden $HIDDEN \
            --heads $heads \
            --dp_size 2 \
            --pp_size 2 \
            --tp_size 2 \
            --gbs $GBS \
            --mbs 128 \
            --seqlen $SEQLEN
    dump_cmd $GBS $LAYERS $heads $HIDDEN $SEQLEN 2 2 2 $((GBS/2/128))
done

echo "GENERATING HIDDEN"
printf "# HIDDEN\n" >> $CMD_FILE
for hidden in "${hidden_sizes[@]}"; do
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy dp \
            --layers $LAYERS \
            --hidden $hidden \
            --heads $HEADS \
            --dp_size 1 \
            --pp_size 1 \
            --tp_size 1 \
            --gbs $GBS \
            --mbs $GBS \
            --seqlen $SEQLEN
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy hybrid \
            --layers $LAYERS \
            --hidden $hidden \
            --heads $HEADS \
            --dp_size 2 \
            --pp_size 2 \
            --tp_size 2 \
            --gbs $GBS \
            --mbs 128 \
            --seqlen $SEQLEN
    dump_cmd $GBS $LAYERS $HEADS $hidden $SEQLEN 2 2 2 $((GBS/2/128))
done

echo "GENERATING SEQLEN"
printf "# SEQLEN\n" >> $CMD_FILE
for seqlen in "${seq_sizes[@]}"; do
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy dp \
            --layers $LAYERS \
            --hidden $HIDDEN_TINY \
            --heads $HEADS \
            --dp_size 1 \
            --pp_size 1 \
            --tp_size 1 \
            --gbs $GBS_TINY \
            --mbs $GBS_TINY \
            --seqlen $seqlen
    PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
        --nproc_per_node=1  \
        gen_model/gen_llama3_default.py --policy hybrid \
            --layers $LAYERS \
            --hidden $HIDDEN_TINY \
            --heads $HEADS \
            --dp_size 2 \
            --pp_size 2 \
            --tp_size 2 \
            --gbs $GBS_TINY \
            --mbs $((GBS_TINY/4)) \
            --seqlen $seqlen
    dump_cmd $GBS_TINY $LAYERS $HEADS $HIDDEN_TINY $seqlen 2 2 2 2
done

