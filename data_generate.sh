cd /home/liuchang/projects/peract-main/RLBench-peract/tools
xvfb-run python dataset_generator.py  \
                            --save_path=/home/liuchang/projects/peract-main/data/high \
                            --image_size=512,512 \
                            --renderer=opengl \
                            --episodes_per_task=1 \
                            --processes=24 \
                            --all_variations=True \
                            