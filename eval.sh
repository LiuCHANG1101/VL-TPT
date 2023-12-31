CUDA_VISIBLE_DEVICES=1 xvfb-run python eval.py \
    rlbench.tasks=[open_drawer,stack_blocks,sweep_to_dustpan_of_size,push_buttons,put_money_in_safe,place_wine_at_rack_location] \
    rlbench.task_name='multi_6T' \
    rlbench.demo_path=/home/liuchang/projects/peract-main/data/val \
    framework.logdir=/home/liuchang/projects/peract-main/logs_lowlangwithprompt \
    framework.csv_logging=True \
    framework.tensorboard_logging=False \
    framework.eval_envs=1 \
    framework.start_seed=1 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=25 \
    framework.eval_type='missing' \
    framework.use_low_lang=True \
    framework.prompt=True \
    rlbench.headless=True