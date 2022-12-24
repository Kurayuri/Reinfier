from ..drlp.DRLP import DRLP
from typing import Callable, Dict, List, Set, Union
from bayes_opt import BayesianOptimization
from ..import alg
from ..import drlp
from ..import util
import subprocess
import os
import pandas as pd
import glob
import re


def choose_curriculum():
    pass


class Reintrainer:
    '''
    Reintrainer: Property Training Framework for Reinforcement Learning
    '''

    def __init__(self, properties: List[DRLP], curriculum_chosen_func: Callable,
                 model_path, verifier: str,
                 train_api: Callable, test_api: Callable,
                 save_dir: str
                 ):
        self.properties = properties
        self.properties_apply = []
        self.curriculum_chosen_func = curriculum_chosen_func
        self.verification_results = []
        self.verifier = verifier
        self.model_path = model_path

# spr
        self.save_dir = save_dir
        self.nproc = 1
        self.seed = 10
        self.real_trace_prob = 0
        self.validation = False
        self.train_trace_file = None
        self.model_select = 'latest'

    def train(self, round: int, step: int):
        for i in range(round):
            # optimizer = BayesianOptimization()
            # optimizer.maximize()
            network = self.model_path

            for property in self.properties:
                # ans = alg.verify(network, property, verifier=self.verifier, to_induct=True)
                ans = (2, False)
                self.verification_results.append(ans)

            training_save_dir = os.path.join(self.save_dir, "bo_{}".format(i))
            self.cur_config_file = os.path.join(
                self.save_dir, "bo_" + str(i) + ".json")

            # cmd = "mpiexec -np {nproc} python src/simulator/train_rl.py " \
            cmd = ". genet/bin/activate; cd Genet; python src/simulator/train_rl.py " \
                "--save-dir {save_dir} --exp-name {exp_name} --seed {seed} " \
                "--total-timesteps {tot_step} " \
                "--randomization-range-file {config_file} " \
                "--real-trace-prob {real_trace_prob}".format(
                    nproc=self.nproc, save_dir=training_save_dir, exp_name="b",
                    seed=self.seed, tot_step=step,
                    config_file=self.cur_config_file,
                    real_trace_prob=self.real_trace_prob)
            if self.model_path:
                cmd += " --pretrained-model-path {}".format(self.model_path)
            if self.validation:
                cmd += " --validation"
            if self.train_trace_file:
                cmd += " --train-trace-file {}".format(self.train_trace_file)
            # subprocess.run(cmd.split(' '))
            # proc = subprocess.run(cmd.split(" "), capture_output=True, text=True)
            proc = subprocess.run(cmd, text=True, shell=True,stdout=subprocess.PIPE)


            dnnv_stdout = proc.stdout
            dnnv_stderr = proc.stderr
            print(dnnv_stderr)
            print("\n")
            print(dnnv_stdout)
            self.model_path = get_model_from(training_save_dir, self.model_select)
            print(self.model_path)
            # assert self.model_path

    def exec_constraint(self, code, x, y):
        exec(code)
        if isinstance(code, str):
            del code
        return locals()

    def reward(self, x, y, reward: float):
        for property in self.properties:
            x = [x]
            y = [y]
            constraint = drlp.parse_drlp_get_constraint(property)
            util.log("## Constraint:\n", constraint.obj)
            util.log(constraint.obj)
            code = drlp.parse_constaint_to_code(constraint)
            violated = self.exec_constraint(code, x, y)[drlp.VIOLATED_ID]
            if violated:
                reward = -4.5
        return reward


def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_model_from(path: str, opt='latest') -> str:
    if opt == 'latest':
        ckpts = list(glob.glob(os.path.join(path, "model_step_*.ckpt.meta")))
        if not ckpts:
            ckpt = ""
        else:
            ckpt = os.path.splitext(natural_sort(ckpts)[-1])[0]

        return ckpt
    elif opt == 'best':
        df = pd.read_csv(os.path.join(path, "validation_log.csv"), sep='\t')
        assert isinstance(df, pd.DataFrame)
        best_idx = df['mean_validation_reward'].argmax()
        best_step = int(df['num_timesteps'][best_idx])
        best_ckpt = os.path.join(path, "model_step_{}.ckpt".format(best_step))

        return best_ckpt
    raise ValueError
