#!/bin/bash
python ppo.py --save_policy --total_timesteps 1e8
python generate_end_policy_data.py 
python ppo.py --get_projections --total_timesteps 1e8