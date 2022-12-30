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
import ast
import astor


def choose_curriculum():
    pass


class Reintrainer:
    '''
    Reintrainer: Property Training Framework for Reinforcement Learning
    '''

    REWARD_FUNC_ID = "reward"
    REWARD_FUNC_PARA_REWARD_ID = "rwd"
    REWARD_FUNC_PARA_VIOLATED_ID = "violated"

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
        self.reward_api = "reward_api.py"

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

            self.generate_constant()
            self.generate_reward()

            self.call_train_api(total_timestep=step)

            self.curr_model_path = self.next_model_path
            util.log("\n## Current model path: \n%s" % self.curr_model_path, level=CONSTANT.INFO)
            break


    def generate_constant(self):
        code = self.get_constraint(self.properties[0])
        with open(f"{self.next_model_path}/{self.reward_api}", "w") as f:
            f.write(code)
        return code

    def generate_reward(self, to_append: bool = True):
        reward_val = -4.5

        ast_root = ast.parse("")
        ast_root.body = [ast.FunctionDef(
            name=self.REWARD_FUNC_ID, decorator_list=[],
            args=ast.arguments(
                args=[ast.arg(arg=self.REWARD_FUNC_PARA_VIOLATED_ID, annotation=None),
                      ast.arg(arg=self.REWARD_FUNC_PARA_REWARD_ID, annotation=None),
                      ], defaults=[], vararg=None, kwarg=None
            ),
            body=[ast.If(
                test=ast.Name(id=self.REWARD_FUNC_PARA_VIOLATED_ID, ctx=ast.Load()), orelse=[],
                body=[ast.Assign(
                    targets=[ast.Name(id=self.REWARD_FUNC_PARA_REWARD_ID, ctx=ast.Store())],
                    value=ast.Constant(value=reward_val),
                    type_comment=None,
                )]),
                ast.Return(value=ast.Name(id=self.REWARD_FUNC_PARA_REWARD_ID, ctx=ast.Load()))])]
        code = astor.to_source(ast_root)
        code = "\n" + code
#         code = '''
# def reward(violated,rwd):
#     if violated:
#         rwd = %f
#     return rwd
#         ''' % (reward_val)
        mode = "a+"
        if not to_append:
            mode = "w"
        with open(f"{self.next_model_path}/{self.reward_api}", mode) as f:
            f.write(code)

    def call_train_api(self, **kwargs):
        if isinstance(self.train_api, Callable):
            kwargs["next_model_path"] = self.next_model_path
            kwargs["reward_api"] = self.reward_api

            if self.curr_model_path:
                kwargs["curr_model_path"] = self.curr_model_path
            self.train_api(**kwargs)

        elif isinstance(self.train_api, str):
            cmd = self.train_api + " " \
                f"--total_timestep  {kwargs['total_timestep']} " \
                f"--next_model_path {self.next_model_path} " \
                f"--reward_api      {self.reward_api} " \
                ""

            if self.curr_model_path:
                cmd += f"--curr_model_path {self.curr_model_path} "

            util.log(cmd, level=CONSTANT.INFO)
            # with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            #     for line in process.stdout:
            #         util.log(line.decode('utf8'))

            # proc = subprocess.run(cmd.split(" "), capture_output=True, text=True)
            proc = subprocess.run(cmd, shell=True, capture_output=False)
            # dnnv_stdout = proc.stdout
            # dnnv_stderr = proc.stderr
            # print(dnnv_stderr)
            # print("\n")
            # print(dnnv_stdout)

    def get_constraint(self, property):   # TODO
        constraint = drlp.parse_drlp_get_constraint(property)
        util.log("## Constraint:\n", constraint.obj)
        util.log(constraint.obj)
        code = drlp.parse_constaint_to_code(constraint)
        return code

    def exec_constraint(self, code, x, y):
        exec(code)
        return locals()[drlp.IS_VIOLATED_ID](x, y)

    def get_next_model_path(self, rnd: int):
        path = self.save_path + "/" + "bo_%d" % rnd
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            util.log(e, level=CONSTANT.ERROR)
        return path

    def reward(self, x, y, reward: float):
        for property in self.properties:
            x = [x]
            y = [y]
            code = self.get_constraint(property)
            violated = self.exec_constraint(code, x, y)
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
