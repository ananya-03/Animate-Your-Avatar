import numpy as np
from brax import envs
from brax.io import html
import streamlit.components.v1 as components
import streamlit as st
import glob

from diffmimic.utils.io import deserialize_qp
from diffmimic.mimic_envs import register_mimic_env

register_mimic_env()

file_mapping = {
#     'A person moves both hands': 'motion_samples/S0eval_traj_best.npy',
    'A person stretches their arms to the left and right ': 'logs/exp_200_200_0.0003_1000_0.3_0_smpl_threshold_0.2_0.05_False_True_60_60_600_True_0.02_1000_0.01_sample01_rep02_smpl_params/eval_traj_best.npy',
    'A man stands prepared then suddenly takes an abrupt step back and regains his stance': 'logs/exp_200_200_0.0003_1000_0.3_0_smpl_threshold_0.2_0.05_False_True_60_60_600_True_0.02_1000_0.01_sample02_rep02_smpl_params/eval_traj_best.npy',
    'A person takes a long stride forward with their right leg': 'logs/exp_200_200_0.0003_1000_0.3_0_smpl_threshold_0.2_0.05_False_True_60_60_600_True_0.02_1000_0.01_sample03_rep02_smpl_params/eval_traj_best.npy',
    'A person jumps': 'logs/exp_200_200_0.0003_1000_0.3_0_smpl_threshold_0.2_0.05_False_True_60_60_600_True_0.02_1000_0.01_sample04_rep02_smpl_params/eval_traj_best.npy',
    'A person does “Namaste': 'logs/exp_200_200_0.0003_1000_0.3_0_smpl_threshold_0.2_0.05_False_True_60_60_600_True_0.02_1000_0.01_sample05_rep02_smpl_params/eval_traj_best.npy',
    'A person walks backwards': 'logs/exp_200_200_0.0003_1000_0.3_0_smpl_threshold_0.2_0.05_False_True_60_60_600_True_0.02_1000_0.01_sample06_rep02_smpl_params/eval_traj_best.npy',
    'A person paces from left to right': 'logs/exp_200_200_0.0003_1000_0.3_0_smpl_threshold_0.2_0.05_False_True_60_60_600_True_0.02_1000_0.01_sample07_rep02_smpl_params/eval_traj_best.npy',
    'A person walks back and forth': 'logs/exp_200_200_0.0003_1000_0.3_0_smpl_threshold_0.2_0.05_False_True_60_60_600_True_0.02_1000_0.01_sample08_rep02_smpl_params/eval_traj_best.npy',
    'A person with difficulty falls and kneels': 'logs/exp_200_200_0.0003_1000_0.3_0_smpl_threshold_0.2_0.05_False_True_60_60_600_True_0.02_1000_0.01_sample09_rep02_smpl_params/S9_eval_traj_best.npyy',
    
}

selected_file = None

text_input = st.text_input('Enter a motion', value='')
if text_input:
    if text_input not in file_mapping:
        st.warning('Motion not found')
    else:
        selected_file = file_mapping[text_input]
        st.success(f'Selected file: {selected_file}')

if selected_file:
    demo_traj = np.load(selected_file)

    if len(demo_traj.shape) == 3:
        demo_traj = demo_traj[:, 1]  # vis env 0

    init_qp = deserialize_qp(demo_traj[0])
    demo_qp = [deserialize_qp(demo_traj[i]) for i in range(demo_traj.shape[0])]

    env = envs.create(env_name='humanoid_mimic',
                      system_config='smpl',
                      reference_traj=demo_traj,
                      cycle_len = 1)
    try:
        components.html(html.render(env.sys, demo_qp), height=500)
    except Exception as e:
        st.write(f"Error: {e}")
#     components.html(html.render(env.sys, demo_qp), height=500)
    
def deserialize_qp(qp_bytes):
    qp = np.loads(qp_bytes)
    if isinstance(qp, tuple):
        qp = tuple(qp_i.reshape(-1) for qp_i in qp)
    else:
        qp = qp.reshape(-1)
    print(qp.shape)  # add this line to print the shape of the object
    return qp    
