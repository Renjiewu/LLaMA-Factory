# FILEPATH='output/qwen2_vl-7b/lora/sft-default'
###
 # @Author: moyueheng moyueheng@126.com
 # @Date: 2024-12-20 22:50:50
 # @LastEditors: moyueheng moyueheng@126.com
 # @LastEditTime: 2024-12-22 15:00:13
 # @FilePath: /LLaMA-Factory/run/upload.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# FILEPATH='output/qwen2_vl-7b/lora/sft-default-split-img'
# FILEPATH='output/qwen2_vl-7b/lora/sft-default-split-dia'
# FILEPATH='output/qwen2_vl-7b/lora/sft-r64-split-img-noise'
# FILEPATH='output/qwen2_vl-7b/lora/sft-r64-split-img-noise-merge'
# FILEPATH='output/qwen2_vl-7b/lora/sft-r64-split-dia-noise'
# FILEPATH='output/qwen2_vl-7b/lora/sft-r64-split-img-noise-epoch10'

# FILEPATH='output/qwen2_vl-7b/lora/sft-r128-split-img-noise'
# FILEPATH='output/qwen2_vl-7b/lora/sft-r128-split-dia-noise'

# FILEPATH='output/qwen2_vl-7b/lora/sft-infer-split-dia'
# FILEPATH='output/qwen2_vl-7b/lora/sft-infer-split-img-noise'
# FILEPATH='output/qwen2_vl-7b/lora/sft-infer-split-img-noise-merge'

# FILEPATH='output/qwen2_vl-7b/lora/sft-infer-split-img-noise-epoch10'
# FILEPATH='output/qwen2_vl-7b/lora/sft-infer-split-dia-noise'

# FILEPATH='output/qwen2_vl-7b/lora/sft-infer-cot-base-img'
# FILEPATH='output/qwen2_vl-7b/lora/sft-infer-cot-base-dia'

# FILEPATH='output/qwen2_vl-7b/lora/sft-img-cot-r64-ep10'
# FILEPATH='output/qwen2_vl-7b/lora/sft-dia-cot-r64-ep10'
# FILEPATH='output/qwen2_vl-7b/lora/sft-img-cot-r128-ep10'
# FILEPATH='output/qwen2_vl-7b/lora/sft-dia-cot-r128-ep10'

# FILEPATH='output/qwen2_vl-7b/lora/sft-infer-cot-img-eval'
# FILEPATH='output/qwen2_vl-7b/lora/sft-infer-cot-dia-eval'

# FILEPATH='output/qwen2_vl-7b/lora/lora_r64_img_ep3'
# FILEPATH='output/qwen2_vl-7b/lora/lora_r64_dia_ep3'
# FILEPATH='output/qwen2_vl-7b/dora/dora_r64_img_ep3'
# FILEPATH='output/qwen2_vl-7b/dora/dora_r64_dia_ep3'

# FILEPATH='output/qwen2_vl-7b/dora/dora_r32_ep3_explain'
# FILEPATH='output/qwen2_vl-7b/dora/dora_r32_ep3_explainv2'


# FILEPATH='examples/www2025'
# FILEPATH='examples/www2025_eval'
# FILEPATH='data/dataset_op.py'
# FILEPATH='data/fix_label.py'
# FILEPATH='data/dataset_info.json'
# FILEPATH='data/mire.tar.gz'
# FILEPATH='data/mire_eval.tar.gz'
# FILEPATH='data/mire_img_noise.tar.gz'
# FILEPATH='run'
# FILEPATH='scripts'
# FILEPATH='data/mire_enhance'

# FILEPATH='data/mire_normal.tar.gz'

# FILEPATH='run/deploy-qwq'
FILEPATH='run/embed_model'

dav -u admin -p admin --endpoint-url http://206.237.7.51:6081/remote.php/dav/files/admin sync $FILEPATH dav://test/www2025/$FILEPATH

# dav -u admin -p admin --endpoint-url http://206.237.7.51:6081/remote.php/dav/files/admin sync dav://test/www2025/$FILEPATH $FILEPATH

# pip install webdav4