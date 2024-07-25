#!/bin/bash
python ppo.py --save_traj --total_timesteps 1e8
python craftax/llc_estimation.py