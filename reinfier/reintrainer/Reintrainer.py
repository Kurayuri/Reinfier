from ..drlp.DRLP import DRLP
from ..nn.NN import NN
from ..import alg
from ..import drlp
from ..import util
from ..import CONSTANT
from typing import Callable, Dict, List, Set, Union, Tuple
from bayes_opt import BayesianOptimization
import pandas as pd
import subprocess
import inspect
import astor
import glob
import ast
import os
import re


def choose_curriculum():
    pass


class Reintrainer:
    '''
    Reintrainer: Property Training Framework for Reinforcement Learning
    '''

    REWARD_FUNC_ID = "reward"
    REWARD_FUNC_PARA_REWARD_ID = "rwd"
    REWARD_FUNC_PARA_VIOLATED_ID = "violated"
    REWARD_API_FILENAME = "reward_api.py"

    def __init__(self, properties: List[DRLP],
                 train_api: Union[Callable, str, Tuple[str, str]],
                 save_path: str,
                 verifier: str,
                 onnx_filename: str = "model.onnx",
                 init_model_dirpath: str = None,
                 reward_api_type: str = None,
                 test_api: Union[Callable, str, Tuple[str, str]] = None,
                 curriculum_chosen_func: Callable = None,
                 ):
        self.properties = properties
        self.properties_apply = []
        self.curriculum_chosen_func = curriculum_chosen_func
        self.verification_results = []
        self.verifier = verifier

        self.init_model_dirpath = init_model_dirpath
        self.curr_model_dirpath = init_model_dirpath
        self.next_model_dirpath = None
        self.onnx_filename = onnx_filename
        if self.curr_model_dirpath is not None:
            self.curr_model_path = os.path.join(self.curr_model_dirpath, self.onnx_filename)
        else:
            self.curr_model_path = None

        self.save_path = save_path
        self.model_select = 'latest'

        self.round = -1
        self.train_api = train_api
        self.test_api = test_api
        if reward_api_type:
            if reward_api_type == "Callable":
                self.reward_api = self.RewardAPI
            elif reward_api_type == "str":
                self.reward_api = self.REWARD_API_FILENAME
        else:
            if isinstance(self.train_api, Callable):
                self.reward_api = self.RewardAPI
            elif isinstance(self.train_api, str) or \
                    isinstance(self.train_api, Tuple):
                self.reward_api = self.REWARD_API_FILENAME

    def train(self, round: int, cycle: int):
        for self.round in range(self.round + 1, self.round + 1 + round):
            util.log_prompt(4)
            util.log("*" * 20 + " Round %d " % self.round + "*" * 20, level=CONSTANT.WARNING)

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
            self.next_model_dirpath = self.make_next_model_dir(self.round)

            self.generate_constant()
            self.generate_reward()

            self.call_train_api(total_cycle=cycle)

            self.curr_model_dirpath = self.next_model_dirpath
            util.log("\n## Current model dirpath: \n%s" % self.curr_model_dirpath, level=CONSTANT.INFO)

            # %% Test
            util.log("########## Testing Part ##########\n", level=CONSTANT.INFO)
            self.call_test_api()

    def generate_constant(self):
        code = self.get_constraint(self.properties[0])
        with open(os.path.join(self.next_model_dirpath, self.reward_api), "w") as f:
            f.write(code)
        return code

    def generate_reward(self, to_append: bool = True):

        # %% Generate reward from AST
        # reward_val = -4.5
        # ast_root = ast.parse("")
        # ast_root.body = [ast.FunctionDef(
        #     name=self.REWARD_FUNC_ID, decorator_list=[],
        #     args=ast.arguments(
        #         args=[ast.arg(arg=self.REWARD_FUNC_PARA_VIOLATED_ID, annotation=None),
        #               ast.arg(arg=self.REWARD_FUNC_PARA_REWARD_ID, annotation=None),
        #               ], defaults=[], vararg=None, kwarg=None
        #     ),
        #     body=[ast.If(
        #         test=ast.Name(id=self.REWARD_FUNC_PARA_VIOLATED_ID, ctx=ast.Load()), orelse=[],
        #         body=[ast.Assign(
        #             targets=[ast.Name(id=self.REWARD_FUNC_PARA_REWARD_ID, ctx=ast.Store())],
        #             value=ast.Constant(value=reward_val),
        #             type_comment=None,
        #         )]),
        #         ast.Return(value=ast.Name(id=self.REWARD_FUNC_PARA_REWARD_ID, ctx=ast.Load()))])]
        # code = astor.to_source(ast_root)

        # %% Generate reward from reward_func

        code = inspect.getsource(Reintrainer.reward)
        indent = re.search("def", code.split("\n", 1)[0]).span()[0]
        code = "\n".join([line[indent:] for line in code.split("\n")])

        ast_root = ast.parse(code)
        ast_root.body[0].args.args = ast_root.body[0].args.args[1:]
        code = astor.to_source(ast_root)

        code = "\n" + code

        mode = "a+"
        if not to_append:
            mode = "w"
        with open(os.path.join(self.next_model_dirpath, self.reward_api), mode) as f:
            f.write(code)

        return code

    def reward(self, violated, rwd):
        if violated:
            rwd = -4.5
        return rwd

    def RewardAPI(self, x, y, rwd: float):
        for property in self.properties:
            x = [x]
            y = [y]
            code = self.get_constraint(property)
            violated = self.exec_constraint(code, x, y)
            rwd = self.reward(violated, rwd)
        return rwd

    def call_train_api(self, **kwargs):
        util.log("Training...", level=CONSTANT.INFO)

        if isinstance(self.train_api, Callable):
            kwargs["next_model_dirpath"] = self.next_model_dirpath
            kwargs["reward_api"] = self.reward_api
            kwargs["total_cycle"] = kwargs['total_cycle']

            if self.curr_model_dirpath:
                kwargs["curr_model_dirpath"] = self.curr_model_dirpath
            self.train_api(**kwargs)

        elif isinstance(self.train_api, str) or \
                isinstance(self.train_api, Tuple):
            cmd_prefix = self.train_api
            cmd_suffix = ""
            if isinstance(self.train_api, Tuple):
                cmd_prefix, cmd_suffix = self.train_api
            cmd = cmd_prefix + " " \
                f"--total_cycle        {kwargs['total_cycle']} " \
                f"--next_model_dirpath {self.next_model_dirpath} " \
                f"--reward_api         {self.reward_api} " \
                ""

            if self.curr_model_dirpath:
                cmd += f"--curr_model_dirpath {self.curr_model_dirpath} "
            cmd += cmd_suffix

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

    def call_test_api(self, **kwargs):
        util.log("Testing...", level=CONSTANT.INFO)

        if isinstance(self.test_api, Callable):
            kwargs["reward_api"] = self.reward_api
            kwargs["curr_model_dirpath"] = self.curr_model_dirpath
            self.test_api(**kwargs)

        elif isinstance(self.test_api, str) or \
                isinstance(self.test_api, Tuple):
            cmd_prefix = self.test_api
            cmd_suffix = ""
            if isinstance(self.test_api, Tuple):
                cmd_prefix, cmd_suffix = self.test_api
            cmd = cmd_prefix + " " \
                f"--curr_model_dirpath {self.curr_model_dirpath} " \
                f"--reward_api         {self.reward_api} " \
                ""
            
            cmd += cmd_suffix

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

    def make_next_model_dir(self, round: int):
        path = os.path.join(self.save_path, f"round_%03d" % (round))
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            util.log(e, level=CONSTANT.ERROR)
        return path

    def load_model(self, path: str) -> NN:
        return NN(path)

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
