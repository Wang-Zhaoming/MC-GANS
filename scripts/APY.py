import subprocess


common_params = [
    '--dataset', 'APY',
    '--class_embedding', 'att',
    '--attSize', '64',
    '--nz', '64',
    '--att_std', '0.02',
    '--lambda_1', '0.07',
    '--temperature', '0.06',
    '--w_up', '10.0',
    '--nepoch_classifier', '80'
]

search_commands = [
    ['python', 'search.py',
     '--syn_num', '50',
     '--nepoch', '300',
     '--critic_iter', '8'],
    ['python', 'retrain.py',
     '--syn_num', '50',
     '--nepoch', '500',
     '--critic_iter', '5',
     '--original', 'False']
]


for i in range(5):
    for base_command in search_commands:
        full_command = base_command + common_params
        print(f"Running command: {' '.join(full_command)}")
        result = subprocess.run(full_command, check=True)
        print(f"Command completed with return code: {result.returncode}")
