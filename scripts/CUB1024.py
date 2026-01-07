import subprocess


common_params = [
    '--dataset', 'CUB',
    '--class_embedding', 'sent',
    '--attSize', '1024',
    '--nz', '1024',
    '--att_std', '0.02',
    '--lambda_1', '2.5',
    '--temperature', '0.04',
    '--w_up', '10.0',
    '--nepoch_classifier', '150'
]


search_commands = [
    ['python', 'search.py',
     '--syn_num', '50',
     '--nepoch', '500',
     '--save_data', 'True',
     # '--manualSeed', '7044',
     '--critic_iter', '8'],
    ['python', 'retrain.py',
     '--syn_num', '50',
     '--nepoch', '500',
     '--critic_iter', '5',
     '--original', 'True',
     # '--manualSeed', '3405',
     '--save_data', 'True',
     '--lr', '0.0001']
]


for i in range(1):
    for base_command in search_commands:
        full_command = base_command + common_params
        print(f"Running command: {' '.join(full_command)} ")
        result = subprocess.run(full_command, check=True)
        print(f"Command completed with return code: {result.returncode}")