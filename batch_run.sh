

alias python='/opt/conda/bin/python'

prefix_path="/workspace/data/models/hg_models/c-model"
samp_dir="/workspace/data/samples"
gpu_name="8-gpu"

model_name_arr=("gpt_18b_fp16" "gpt_39b_fp16" "gpt_175b_fp16")
samp_arr=("sample_20.txt" "sample_80.txt" "sample_320.txt" "sample_960.txt")
out_len_arr=(20 40 80)

for(( i=0;i<${#model_name_arr[@]};i++)) do
   for(( j=0;j<${#samp_arr[@]};j++)) do
     for(( k=0;k<${#out_len_arr[@]};k++)) do
	ckp_path="${prefix_path}/${model_name_arr[i]}/${gpu_name}";
	samp_file="${samp_dir}/${samp_arr[j]}";
	out_len=${out_len_arr[k]};
	prof_log="${model_name_arr[i]}_${gpu_name}_${j}_${out_len}.qdrep";
	echo nsys profile  -c cudaProfilerApi -f true --stats true  -o ${prof_log} \
 	mpirun -n 8 --allow-run-as-root python examples/pytorch/gpt/multi_gpu_gpt_example_nvtx.py  --tensor_para_size=8 --pipeline_para_size=1 \
        --vocab_file models/gpt2-vocab.json --merges_file models/gpt2-merges.txt --lib_path build/lib/libth_transformer.so --time \
        --ckpt_path ${ckp_path}  --sample_input_file ${samp_file} --output_len ${out_len} --inference_data_type fp16;
	nsys profile  -c cudaProfilerApi -f true --stats true  -o ${prof_log} \
 	mpirun -n 8 --allow-run-as-root python examples/pytorch/gpt/multi_gpu_gpt_example_nvtx.py  --tensor_para_size=8 --pipeline_para_size=1 \
        --vocab_file models/gpt2-vocab.json --merges_file models/gpt2-merges.txt --lib_path build/lib/libth_transformer.so --time \
        --ckpt_path ${ckp_path}  --sample_input_file ${samp_file} --output_len ${out_len} --inference_data_type fp16;
	echo "done";
      done
    done
 done

