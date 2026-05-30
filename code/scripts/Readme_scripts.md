1. pilot.py (M1) - Joint training to verify architectures can learn CIFAR-100
2. run_cl.py (M2) - Full CL grid (3 arch x 3 methods x 3 seeds) - this generates the main experimental data
3. extract.py (M3a) - Reads checkpoints from M2 runs, computes CKA + drift
4. plot.py (M3b) - Reads CSVs from M2 + npz from M3a, generates figures
5. eval_task_il.py - Reads M2 checkpoints, does task-IL oracle eval
6. eval_linear_probe.py - Reads M2 checkpoints, does linear probe eval
7. ablation_hp.py - Runs its own mini-M2 to validate hyperparameters