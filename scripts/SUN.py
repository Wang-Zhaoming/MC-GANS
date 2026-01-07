import subprocess

# 循环搜索训练生成器G和鉴别器D
# False True 约 226 张图像
# 定义要运行的命令
common_params = [
    '--dataset', 'SUN',
    '--class_embedding', 'att',
    '--attSize', '102',
    '--nz', '102',
    '--att_std', '0.02',
    '--lambda_1', '2.5',  # 0.8
    '--temperature', '0.04',
    '--w_up', '10.0',
    '--nepoch_classifier', '80'
]

search_commands = [
    ['python', 'search.py',
     '--syn_num', '30',
     '--nepoch', '300',
     '--critic_iter', '8',
     '--batch_size', '512',
     '--save_data', 'True',
     # '--manualSeed', '4656',
     ],
    ['python', 'retrain.py',
     '--syn_num', '30',
     '--nepoch', '500',
     '--critic_iter', '5',
     '--batch_size', '512',
     # '--manualSeed', '8444',
     '--save_data', 'True',
     '--original', 'False']
]

for i in range(1):
    for base_command in search_commands:
        full_command = base_command + common_params
        print(f"Running command: {' '.join(full_command)}")
        result = subprocess.run(full_command, check=True)
        print(f"Command completed with return code: {result.returncode}")
