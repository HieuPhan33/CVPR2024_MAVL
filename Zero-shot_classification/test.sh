filenames=(chexpert_mavl chexray_mavl covid_mavl rsna_mavl siim_mavl padchest_rare_mavl padchest_seen_mavl padchest_unseen_mavl)
ckpts=(25 28 31 34 37 state)
# model_name=output_masked_text5e-1
model_name=mavl_fp16_40
for filename in "${filenames[@]}"; do
    for ckpt in "${ckpts[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python test.py --config configs/${filename}.yaml --model_path ../checkpoints/${model_name}/checkpoint_${ckpt}.pth
    done
done