# model_name=output_masked_text5e-1
for filename in "${filenames[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python test.py --config configs/${filename}.yaml --model_path ../checkpoints/checkpoint_full_40.pth
done

for filename in "${filenames[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python test.py --config configs/${filename}.yaml --model_path ../checkpoints/checkpoint_full_46.pth
done

for filename in "${filenames[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python test.py --config configs/${filename}.yaml --model_path ../checkpoints/checkpoint_short_37.pth
done

