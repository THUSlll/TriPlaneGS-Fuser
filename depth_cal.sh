# count=0
# for dir in /hdd/u202320081001061/ScanNet-GSReg/train/*/; do
#     count=$((count+1))
#     python render_depth.py --config "config_new.yaml" --source_path "$dir" --exp_name "test"
#     echo -e "\033[32mtrain $dir done ($count)\033[0m"
# done

# count=0
# for dir in /hdd/u202320081001061/ScanNet-GSReg/test/*/; do
#     count=$((count+1))
#     python render_depth.py --config "config_new.yaml" --source_path "$dir" --exp_name "test"
#     echo -e "\033[34mtest $dir done ($count)\033[0m"
# done

count=0
for scene_dir in /hdd/u202320081001061/ScanNet-GSReg/test/*/; do
    for dir in $scene_dir/*/; do
        count=$((count+1))
        python run.py --encoder "vitl" --img-path "$dir/images" --outdir "$dir/mono_depth" --pred-only --grayscale
        echo -e "\033[34mtest $dir done ($count)\033[0m"
    done
done

