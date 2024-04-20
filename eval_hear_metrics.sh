emb_path=$1
task_file=$2
gpu_idx=$3

n=1
while read task; do
    # run tasks one by one
    echo "---------------------------------------------------Task  No. $n : $task"
    n=$((n+1))
    CUDA_VISIBLE_DEVICES=${gpu_idx} python3 -m heareval.predictions.runner ${emb_path}/$task
done < $task_file