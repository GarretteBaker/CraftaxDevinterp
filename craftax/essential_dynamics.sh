#!/bin/bash
python ppo.py --save_policy True --get_projections False --total_timesteps 1e8
python generate_end_policy_data.py 
python ppo.py --save_policy False --get_projections True --total_timesteps 1e8