log_dir=/workspace/data/fast_logs/opt2
mkdir -p ${log_dir}
gpu_logs=${log_dir}/gpu_logs
cpu_logs=${log_dir}/cpu_logs
mkdir -p ${gpu_logs}
mkdir -p ${cpu_logs}

model_dir="/workspace/data/hg_models"
voc_dir="examples/pytorch/gpt/utils/opt_model_configs"

num_gpu=8
arg1="--vocab_file ${voc_dir}/vocab.json --merges_file ${voc_dir}/merges.txt --lib_path build/lib/libth_transformer.so --inference_data_type fp16  --time"

model_name_arr=("opt-66b" "opt-66b" "opt-66b")
input_len=512
out_len=64
bs_arr=(1 8 16)
tp=4
pp=2
num_bs=1
devices="1,2,3,4,5,6,7,8"

for(( i=0;i<${#model_name_arr[@]};i++)) do
      bs=${bs_arr[i]};
      # input_len=${input_len_arr[i]};
      # out_len=${out_len_arr[i]};
      model_name=${model_name_arr[i]};
      ps="ps";
      model_path=${model_dir}/${model_name}/${tp}-gpu;
      gpu_log="${gpu_logs}/${model_name}_${input_len}_${out_len}_${bs}_${num_bs}_${ps}_0_${num_gpu}.qdrep";
      cpu_log="${cpu_logs}/${model_name}_${input_len}_${out_len}_${bs}_${num_bs}_${ps}_0_${num_gpu}.txt";
      cpu_log_org="${cpu_log}.org";
      cmd_pre="mpirun -n ${num_gpu} --allow-run-as-root python "
      exec_path="examples/pytorch/gpt/multi_gpu_gpt_example_org.py"
      args_str="--tensor_para_size=${tp} --pipeline_para_size=${pp} --ckpt_path ${model_path} ${arg1} --sample_input_file samples/sample_960.txt --output_len ${out_len}  --max_batch_size $bs --input_len ${input_len}";
      cmd="${cmd_pre} ${exec_path} $args_str --log-file ${cpu_log_org}";
      echo $cmd;
      CUDA_VISIBLE_DEVICES=${devices} $cmd;
      exec_path="examples/pytorch/gpt/multi_gpu_gpt_example_prof.py"
      sleep 30
      cmd="nsys profile  -c cudaProfilerApi -f true --stats true  -o ${gpu_log} ${cmd_pre} ${exec_path} $args_str --cpu_log_path ${cpu_log}";
      echo $cmd;
      CUDA_VISIBLE_DEVICES=${devices} $cmd;
      echo "done";
      sleep 30
done

