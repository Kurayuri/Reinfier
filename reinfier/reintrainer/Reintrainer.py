from ..drlp.DRLP import DRLP
from ..nn.NN import NN
from typing import Callable, Dict, List, Set, Union
from bayes_opt import BayesianOptimization
from ..import alg
from ..import drlp
from ..import util
from .. import CONSTANT
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
                 init_model_path: str, verifier: str,
                 train_api: Union[Callable, str], test_api: Union[Callable, str],
                 save_path: str, onnx_filename: str = "model.onnx"
                 ):
        self.properties = properties
        self.properties_apply = []
        self.curriculum_chosen_func = curriculum_chosen_func
        self.verification_results = []
        self.verifier = verifier

        self.init_model_path = init_model_path
        self.curr_model_path = init_model_path
        self.next_model_path = None

        self.train_api = train_api
        self.test_api = test_api

        self.onnx_filename = onnx_filename

        self.save_path = save_path
        self.model_select = 'latest'

    def train(self, round: int, step: int):
        for rnd in range(round):
            util.log_prompt(4)
            util.log("*" * 20 + " Round %d " % rnd + "*" * 20, level=CONSTANT.WARNING)

            # optimizer = BayesianOptimization()
            # optimizer.maximize()

            # %% Verify
            util.log("\n########## Verification Part ##########\n", level=CONSTANT.INFO)

            if self.curr_model_path is not None:
                network = self.load_model(self.curr_model_path)

                for property in self.properties:
                    ans = alg.verify(network, property, verifier=self.verifier, to_induct=True)
                    self.verification_results.append(ans)
                util.log("## Verification Results: ", level=CONSTANT.INFO)
                for i in range(len(self.properties)):
                    util.log(self.properties[i], self.verification_results[i], level=CONSTANT.INFO)

            # %% Train
            util.log_prompt(3)
            util.log("########## Training Part ##########\n", level=CONSTANT.INFO)
            self.next_model_path = self.get_next_model_path(rnd)

            self.call_train_api(self.train_api,
                                curr_model_path=self.curr_model_path,
                                next_model_path=self.next_model_path,
                                total_timestep=step,
                                reward_api=self.reward)

            self.curr_model_path = self.next_model_path
            util.log("\n## Current model path: \n%s" % self.curr_model_path, level=CONSTANT.INFO)

    def call_train_api(self, api, **kwargs):
        if isinstance(api, Callable):
            api(kwargs)
        elif isinstance(api, str):

            # cmd = "mpiexec -np {nproc} python src/simulator/train_rl.py " \
            cmd = api
            cmd += " " \
                f"--next_model_path {kwargs['next_model_path']} " \
                f"--total_timestep  {kwargs['total_timestep']} " \
                ""
            # f"--reward_api      {kwargs['reward_api']} "

            if kwargs["curr_model_path"]:
                cmd += f"--curr_model_path {kwargs['curr_model_path']} "

            util.log(cmd, level=CONSTANT.INFO)
            # with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            #     for line in process.stdout:
            #         util.log(line.decode('utf8'))

            # subprocess.run(cmd.split(' '))
            # proc = subprocess.run(cmd.split(" "), capture_output=True, text=True)
            proc = subprocess.run(cmd, shell=True, capture_output=False)
            # dnnv_stdout = proc.stdout
            # dnnv_stderr = proc.stderr
            # print(dnnv_stderr)
            # print("\n")
            # print(dnnv_stdout)

    def exec_constraint(self, code, x, y):
        exec(code)
        if isinstance(code, str):
            del code
        return locals()

    def get_next_model_path(self, rnd: int):
        path = self.save_path + "/" + "bo_%d" % rnd
        try:
            os.mkdir(self.next_model_path)
        except BaseException:
            pass
        return path

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

    def load_model(self, path: str) -> NN:
        return NN(path + "/" + self.onnx_filename)

    # def get_model_from(path: str, opt='latest') -> str:
    #     if opt == 'latest':
    #         ckpts = list(glob.glob(os.path.join(path, "model_step_*.ckpt.meta")))
    #         if not ckpts:
    #             ckpt = ""
    #         else:
    #             ckpt = os.path.splitext(natural_sort(ckpts)[-1])[0]

    #         return ckpt
    #     elif opt == 'best':
    #         df = pd.read_csv(os.path.join(path, "validation_log.csv"), sep='\t')
    #         assert isinstance(df, pd.DataFrame)
    #         best_idx = df['mean_validation_reward'].argmax()
    #         best_step = int(df['num_timesteps'][best_idx])
    #         best_ckpt = os.path.join(path, "model_step_{}.ckpt".format(best_step))

    #         return best_ckpt
    #     raise ValueError


def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
