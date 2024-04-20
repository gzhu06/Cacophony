export LD_LIBRARY_PATH=LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export PATH=/usr/local/cuda-11.2/bin:$PATH

emb_name=/embeddings # cacophony or audiomae

ckpt_path=$1
embed_save_path=$2
task_data_root=$3
task_file=$4
gpu_idx=$5
sample_rate=$6

n=1
while read line; do
    a=( $line )
    task=${task_data_root}/${a[0]}
    batch_size=${a[1]}
    max_audio_len=${a[2]}

    # run tasks one by one
    echo "---------------------------------------------------Task  No. $n : $task ${batch_size}"
    n=$((n+1))
    echo ${ckpt_path} ${task} ${emb_name} ${batch_size} ${max_audio_len}
    CUDA_VISIBLE_DEVICES=${gpu_idx} time python3 -m heareval.embeddings.runner --model-path ${ckpt_path} --tasks-dir $task --embeddings-dir ${embed_save_path} --embedding-name ${emb_name} --batch-size ${batch_size} --max-audio-len ${max_audio_len} --sample-rate ${sample_rate}
    
done < $task_file