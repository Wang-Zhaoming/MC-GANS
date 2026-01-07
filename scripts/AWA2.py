import subprocess

common_params = [
    '--dataset', 'AWA2',
    '--class_embedding', 'att',
    '--attSize', '85',
    '--nz', '85',
    '--att_std', '0.02',
    '--lambda_1', '0.005',  # 0.005
    '--temperature', '0.04',  # 0.04
    '--w_up', '10.0',
    '--nepoch_classifier', '80'
]

search_commands = [
    ['python', 'search.py',
     '--syn_num', '100',
     '--nepoch', '300',
     # '--manualSeed', '2185',
     '--save_data', 'True',
     '--test_data', 'True',
     '--batch_size', '256',
     '--critic_iter', '8'],
    ['python', 'retrain.py',
     '--syn_num', '100',
     '--save_data', 'True',
     '--nepoch', '500',
     '--critic_iter', '5',
     # '--manualSeed', '2412',
     '--batch_size', '256',
     '--original', 'False']
]

for i in range(1):
    for base_command in search_commands:
        full_command = base_command + common_params
        print(f"Running command: {' '.join(full_command)}")
        result = subprocess.run(full_command, check=True)
        print(f"Command completed with return code: {result.returncode}")

