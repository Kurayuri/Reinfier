from ..util.TimerGroup import TimerGroup
from ..drlp.Feature import Dynamic, Static
from ..drlp.DRLP import DRLP
from ..nn.NN import NN
from ..import interpretor
from ..import CONSTANT
from ..import Setting
from ..import drlp
from ..import util
from ..import alg
from ..import nn
from typing import Callable, Dict, List, Set, Union, Tuple
from collections import namedtuple
# from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import subprocess
import inspect
import astor
import time
import glob
import json
import ast
import os
import re

def choose_curriculum():
    pass


class Reintrainer:
    '''
    Reintrainer: Property Training Framework for Reinforcement Learning
    '''

    REWARD_FUNC_ID = "get_reward"
    REWARD_FUNC_PARA_REWARD_ID = "reward"
    REWARD_FUNC_PARA_VIOLATED_ID = "violated"
    REWARD_API_FILENAME = "reward_api.py"
    NORM_P1 = 1
    NORM_P2 = 2

    def __init__(self, properties: List[DRLP],
                 train_api: Union[Callable, str, Tuple[str, str]],
                 save_dirpath: str,
                 verifier: str,
                 onnx_filename: str = "model.onnx",
                 init_model_dirpath: str = None,
                 round_exsited: int = -1,
                 reward_api_type: str = None,
                 test_api: Union[Callable, str, Tuple[str, str]] = None,
                 test_log_filename: str = "test.log",
                 verify_log_filename: str = "verify.log",
                 time_log_filename: str = "time.log",
                 curriculum_api: Callable = None
                 ):
        # Proeprty
        self.properties = properties
        self.properties_apply = []
        self.verifier = verifier

        # External API
        self.curriculum_api = curriculum_api
        self.train_api = train_api
        self.test_api = test_api

        # Internal API
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

        # Resume
        self.save_dirpath = save_dirpath
        self.round = round_exsited
        self.round_exsited = round_exsited
        self.resumed = False
        if self.round != -1 and (not init_model_dirpath):
            self.resumed = True
            init_model_dirpath = self.make_next_model_dir(self.round)

        try:
            os.makedirs(self.save_dirpath)
        except BaseException:
            if not self.resumed:
                if not util.lib.confirm_input(
                        f"Detected save_path {self.save_dirpath} already exists, but Reinttrainer does not start in resuming training mode. Force to continue ?", itype=CONSTANT.INTERACTIVE_ITYPE_y_or_N):
                    raise Exception("Exit")

        # Path Filename
        self.onnx_filename = onnx_filename
        self.test_log_filename = test_log_filename
        self.verify_log_filename = verify_log_filename
        self.time_log_filename = time_log_filename

        self.init_model_dirpath = init_model_dirpath
        self.curr_model_dirpath = init_model_dirpath
        self.next_model_dirpath = None

        self.curr_model_path = os.path.join(self.curr_model_dirpath, self.onnx_filename) \
            if self.curr_model_dirpath is not None else None

        # Logger
        self.global_log_mode = "a" if self.resumed else "w"
        if self.test_api:
            self.logger_test = open(os.path.join(self.save_dirpath, self.test_log_filename), self.global_log_mode)
        self.logger_verify = open(os.path.join(self.save_dirpath, self.verify_log_filename), self.global_log_mode)
        # Timer
        self.timerGroup = TimerGroup(["Generation", "Training", "Verification", "Testing"])

        # DYN
        self.property_dynamics = [None for i in range(len(self.properties))]
        self.property_statics = [None for i in range(len(self.properties))]

        # self.model_select = 'latest'

    def train(self, round: int, cycle: int):

        # %% Init
        util.log_prompt(4, "Init", style=CONSTANT.STYLE_CYAN)
        util.log("Model: ", self.curr_model_path, level=CONSTANT.INFO)


        logLevel = Setting.set_LogLevel(CONSTANT.ERROR)
        init_verification_results = self.call_verify_api()
        init_test_result = self.call_test_api()
        Setting.set_LogLevel(logLevel)
        util.log_prompt(3, "Summary Part", style=CONSTANT.STYLE_YELLOW)
        util.log(self.curr_model_dirpath, level=CONSTANT.INFO)
        self.log_result(init_verification_results, init_test_result)

        # %% Start
        self.timerGroup.reset()
        self.timerGroup.start()
        for self.round in range(self.round + 1, self.round + 1 + round):
            util.log_prompt(4, "Round %03d" % self.round, style=CONSTANT.STYLE_CYAN)

            self.next_model_dirpath = self.make_next_model_dir(self.round)

            # %% Select
            # optimizer = BayesianOptimization()
            # optimizer.maximize()

            # %% Train
            util.log_prompt(3, "Generation Part", style=CONSTANT.STYLE_BLUE)

            # % Generation
            self.generate_constant()
            self.generate_reward()

            self.timerGroup.switch()
            # % Train
            util.log_prompt(3, "Training Part", style=CONSTANT.STYLE_BLUE)
            self.call_train_api(total_cycle=cycle)

            self.curr_model_dirpath = self.next_model_dirpath
            self.curr_model_path = os.path.join(self.curr_model_dirpath, self.onnx_filename)
            util.log("## Current model dirpath: \n%s" % self.curr_model_dirpath, level=CONSTANT.INFO)

            self.timerGroup.switch()
            # %% Verify
            util.log_prompt(3, "Verification Part", style=CONSTANT.STYLE_MAGENTA)
            util.log("Try to verify model:", self.curr_model_path, level=CONSTANT.INFO)

            self.verification_results = self.call_verify_api()

            util.log("\n## Verification Results: ", level=CONSTANT.WARNING)
            self.log_result(self.verification_results, verbose=True)

            self.timerGroup.switch()
            # %% Test
            util.log_prompt(3, "Testing Part", style=CONSTANT.STYLE_GREEN)
            test_result = self.call_test_api()

            self.timerGroup.switch()
            # %% Time
            util.log_prompt(3, "Summary Part", style=CONSTANT.STYLE_YELLOW)
            self.timerGroup.dump(os.path.join(self.save_dirpath, "time.log"))

            times = self.timerGroup.get().values()

            util.log("Round %03d:" % self.round, level=CONSTANT.INFO)
            util.log(self.curr_model_dirpath, level=CONSTANT.INFO)
            self.log_result(self.verification_results, test_result)
            util.log("Reward Generation Time: %.2f s    Training Time: %.2f s    Test Time: %.2f s Verification Time: %.2f s" % (tuple(times)), level=CONSTANT.INFO)
            util.log("Round Time: %.2f s    Total Time: %.2f s" % (sum(times), self.timerGroup.now()[2]), level=CONSTANT.INFO)

        # %% Final
        self.timerGroup.stop()
        self.timerGroup.dump(os.path.join(self.save_dirpath, "time.log"))

        util.log_prompt(4, "Final", style=CONSTANT.STYLE_CYAN)
        util.log("Model: ", self.curr_model_path, level=CONSTANT.INFO)
        util.log("Total Time: %.2f s" % (self.timerGroup.now()[2]), level=CONSTANT.INFO)

        util.log("From:", level=CONSTANT.INFO, style=CONSTANT.STYLE_RED)
        self.log_result(init_verification_results, init_test_result)

        util.log("\nTo:", level=CONSTANT.INFO, style=CONSTANT.STYLE_GREEN)
        self.log_result(self.verification_results, test_result)

    def generate_constant(self):
        # TODO
        for i in range(len(self.properties)):
            code,dynamics,statics = self.get_constraint(self.properties[i])
            self.property_dynamics[i] = dynamics
            self.property_statics[i] = statics
        with open(os.path.join(self.next_model_dirpath, self.reward_api), "w") as f:
            f.write(code)
        return code


    def generate_reward(self, to_append: bool = True):
        # statics = {
        #     0: Static(0, 1)
        # }
        # dynamics = {
        #     1: Dynamic(-0.1, 0.1, lower_rho=1, upper_rho=2, weight=1),
        #     2: Dynamic(-0.2, 0.2, lower_rho=10,upper_rho= 20,weight= 3)
        # }

        dynamics = self.property_dynamics[0][0]
        statics = self.property_statics[0][0]

        mode = "a+"
        if not to_append:
            mode = "w"

        to_weight = False
        to_measure = True

        if self.curr_model_dirpath and to_measure:
            network = NN(self.curr_model_path)
            for idx, dynamic in dynamics.items():
                dynamic.lower_rho = self.measure_rho(network, dynamics,statics, idx, True)
                dynamic.upper_rho = self.measure_rho(network, dynamics,statics,  idx, False)
                dynamic.weight = (dynamic.lower_rho + dynamic.upper_rho) / 2
        else:
            for idx, dynamic in dynamics.items():
                dynamic.lower_rho = 1
                dynamic.upper_rho = 1
                dynamic.weight = 1

        # if self.curr_model_dirpath and impan:
        #     network = self.load_model(self.curr_model_path)
        #     imp_lower=self.analyze_interpretor_importance(network,[[0,a_low, b_low]],0)
        #     imp_upper=self.analyze_interpretor_importance(network,[[1,a_upp, b_upp]],1)
        #     imp_a=(imp_lower[0]+imp_upper[0])/2
        #     imp_b=(imp_lower[1]+imp_upper[1])/2
        util.log("Importance Weight:", {idx:dynamic.weight for idx,dynamic in dynamics.items()}, level=CONSTANT.WARNING, style=CONSTANT.STYLE_RED)

        # imp_a=1
        # imp_b=5

#         code = f'''
# def is_violated(x, y):
#     violated = False
#     occurred = False
#     if x[0][0] <= 1 and x[0][0] >= 0 and x[0][2] <= 0.2 and x[0][2] >= -0.2 and x[0][1]>=-0.1 and x[0][1]<=0.1:
#         occurred = True
#         if not (y[0][0] < 1 and y[0][0] >= -1):
#             violated = True
#     return occurred, violated

# '''


# def reward(violated, rwd, x, y):
#     p1=1
#     p2=2
#     def dist(lower, upper, val, mid):
#         if val>mid:
#             return ((upper-val)/(upper-mid))**p1
#         else:
#             return ((val-lower)/(mid-lower))**p1


#     if violated:
#         if {wt}:
#             dist_x0 = dist({a_low},{a_upp},x[0][1],{mid_a})
#             dist_x1 = dist({b_low},{b_upp},x[0][2],{mid_b})
#             dist_y = 1
#             rwd = rwd - 4*dist_y * (({imp_a}*dist_x0**2 + {imp_b}*dist_x1**2)**(1/p2))
#         else:
#             rwd = rwd
#     return rwd
# '''

        dist_srcs = []

        sum_weight = 0
        for idx, dynamic in dynamics.items():
            mid = (dynamic.lower_rho * dynamic.lower + dynamic.upper_rho * dynamic.upper) / (dynamic.lower_rho + dynamic.upper_rho)
            src = f"dists_x[{idx}] = {dynamic.weight}*dist(x[0][{idx}],{dynamic.lower},{dynamic.upper},{mid})"
            dist_srcs.append(src)

            sum_weight += dynamic.weight

        get_reward_code = f'''
def get_reward(violated, reward, x ,y):
    p1 = {self.NORM_P1}
    p2 = {self.NORM_P2}

    def dist(val,lower, upper, mid):
        if val > upper or val < lower:
            return 0

        if val > mid:
            return ((upper-val)/(upper-mid))**p1
        else:
            return ((val-lower)/(mid-lower))**p1

    if violated:
        dists_x = dict()
''' + "".join(["        " + srs + "\n" for srs in dist_srcs]) + f'''
        sum_1 = sum([dist**p2 for dist in dists_x.values()])
        sum_2 = {sum_weight}
        Dist_x = (sum_1)**(1/p2)/sum_2
        Fs = - 1 * Dist_x
        reward = reward + Fs
    else:
        reward = reward
    return reward
'''

        with open(os.path.join(self.next_model_dirpath, self.reward_api), mode) as f:
            f.write(get_reward_code)

        return get_reward_code

    def measure_rho(self, network: NN, dynamics: List[Dynamic], statics:List[Static], idx: int, is_lower: bool):
        util.log("Dynamic",idx,level=CONSTANT.INFO)
        logLevel = Setting.set_LogLevel(CONSTANT.CRITICAL)

        input_size, output_size = network.size()

        x_base = [[0 for i in range(input_size)]]
        for idx, dynamic in dynamics.items():
            x_base[0][idx] = dynamic.lower if is_lower else dynamic.upper
        for idx, static in statics.items():
            x_base[0][idx] = static.lower if is_lower else static.upper    
        
        y_base = nn.run_onnx(network, np.array(x_base, dtype=np.float32))

        x_eps_id = "x_eps"
        kwargs = {
            x_eps_id: {"lower_bound": 0,
                       "upper_bound": 0.2,
                       "precise": 0.01,
                       "method": "binary", },
        }


        values = {
            "y_eps": 1e-0,
            "y_base": float(y_base[0][0])
        }

        bound = [str(x_base[0][i]) for i in range(input_size)]

        if is_lower == 0:
            bound_lower = ",".join(bound)
            bound[idx] = bound[idx] + "+" + x_eps_id
            bound_upper = ",".join(bound)
        else:
            bound_upper = ",".join(bound)
            bound[idx] = bound[idx] + "-" + x_eps_id
            bound_lower = ",".join(bound)

        src = f'''
@Pre
x_size = {input_size}
y_size = {output_size}
[[{bound_lower}]] <= x[0] <= [[{bound_upper}]]


@Exp
y_base-y_eps<=y[0][0]<=y_base+y_eps
'''
        property = DRLP(src).set_values(values)
        bps = alg.search_break_points(network, property, kwargs, 0.01, self.verifier, k_max=1, to_induct=False)
        inline_bps, inline_bls = interpretor.analyze_break_points(bps)
        answer,__,__ = interpretor.answer_importance_analysis(inline_bps)
        Setting.set_LogLevel(logLevel)

        return answer[0] if answer is not None else 1

#         kwargs = {
#             "x_eps": {"lower_bound": 0,
#                         "upper_bound": 0.5,
#                         "precise": 0.05,
#                         "method": "binary", },
#         }

#         if lu==0:
#             src = '''
# @Pre
# x_size = 3
# y_size = 1
# [[0,a,b]] <= x[0] <= [[0,a ,b+ x_eps]]


# @Exp
# y[0][y_original]-y_eps<=y[0][0]<=y[0][y_original]+y_eps
# '''
#         else:
#             src = '''
# @Pre
# x_size = 3
# y_size = 1
# [[1,a,b-x_eps]] <= x[0] <= [[1,a,b]]

# @Exp
# y[0][y_original]-y_eps<=y[0][0]<=y[0][y_original]+y_eps
# '''

#         property=DRLP(src).set_values(values)
#         bps = alg.search_break_points(network, property, kwargs, 0.01, self.verifier, k_max=1, to_induct=False)
#         inline_bps, inline_bls = interpretor.analyze_break_points(bps)
#         answer = interpretor.answer_importance_analysis(inline_bps)
#         answers.append(answer[0])

#         absolute_importance=[]

        # for answer in answers:
        #     if not answer:
        #         answer = [1]
        #     absolute_importance.append(1/answer[0])
        # imp_sum=sum(absolute_importance)
        # relative_importace = [imp/imp_sum for imp in absolute_importance]
        # util.log("Absolute Importance:",absolute_importance,level=CONSTANT.WARNING,style=CONSTANT.STYLE_RED)
        # util.log("Relative Importance:",relative_importace,level=CONSTANT.WARNING,style=CONSTANT.STYLE_RED)


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
            proc = subprocess.run(cmd, shell=True, capture_output=False)

    def call_test_api(self, **kwargs):
        if self.test_api is None:
            return None
        if self.curr_model_dirpath is None:
            return None

        util.log("Testing...", level=CONSTANT.INFO)
        result = None

        if isinstance(self.test_api, Callable):
            # kwargs["reward_api"] = self.reward_api
            kwargs["curr_model_dirpath"] = self.curr_model_dirpath
            result = self.test_api(**kwargs)

        elif isinstance(self.test_api, str) or \
                isinstance(self.test_api, Tuple):
            cmd_prefix = self.test_api
            cmd_suffix = ""
            if isinstance(self.test_api, Tuple):
                cmd_prefix, cmd_suffix = self.test_api
            cmd = cmd_prefix + " " \
                f"--curr_model_dirpath {self.curr_model_dirpath} " \
                ""
            # f"--reward_api         {self.reward_api} " \

            cmd += cmd_suffix

            util.log(cmd, level=CONSTANT.INFO)
            # with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            #     for line in process.stdout:
            #         util.log(line.decode('utf8'))

            # proc = subprocess.run(cmd.split(" "), capture_output=True, text=True)
            proc = subprocess.run(cmd, shell=True, capture_output=False)

            result = self.load_test_result(os.path.join(self.curr_model_dirpath, self.test_log_filename))
            # dnnv_stdout = proc.stdout
            # dnnv_stderr = proc.stderr
            # print(dnnv_stderr)
            # print("\n")
            # print(dnnv_stdout)

        self.logger_test.write(f"Round {self.round:03d} Test: mean_reward:{result['mean_reward']:.2f} +/- {result['std_reward']:.2f}\n")
        self.logger_test.flush()
        return result

    def call_verify_api(self, model_path: str = ""):
        verification_results = []

        if not model_path:
            model_path = self.curr_model_path

        if self.curr_model_path is not None:
            network = NN(self.curr_model_path)

            for property in self.properties:
                # TODO
                # ans = alg.verify(network, property, verifier=self.verifier, to_induct=True)
                ans = alg.verify(network, property, verifier=self.verifier, to_induct=False, k_max=1)
                verification_results.append(ans)

            anss = {'%02d' % i: verification_results[i][0] for i in range(len(self.properties))}
            self.logger_verify.write(f"Round {self.round:03d} Verify: {anss}\n")
            self.logger_verify.flush()
        return verification_results

    def make_next_model_dir(self, round: int):
        path = os.path.join(self.save_dirpath, f"round_%03d" % (round))
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            util.log(e, level=CONSTANT.ERROR)
        return path

    # def load_model(self, path: str) -> NN:
    #     return NN(path)

    def load_test_result(self, path: str) -> Dict:
        result = None
        try:
            f = open(path, "r")
            try:
                result = json.load(f)
            except Exception:
                util.log(f"Invalid test result file at {path}.", level=CONSTANT.WARNING, style=CONSTANT.STYLE_YELLOW)
        except Exception:
            util.log(f"Could not open test result file at {path}.", level=CONSTANT.WARNING, style=CONSTANT.STYLE_YELLOW)

        return result

    def get_model_from(path: str, opt='latest') -> str:
        pass
        # if opt == 'latest':
        #     ckpts = list(glob.glob(os.path.join(path, "model_step_*.ckpt.meta")))
        #     if not ckpts:
        #         ckpt = ""
        #     else:
        #         ckpt = os.path.splitext(natural_sort(ckpts)[-1])[0]

        #     return ckpt
        # elif opt == 'best':
        #     df = pd.read_csv(os.path.join(path, "validation_log.csv"), sep='\t')
        #     assert isinstance(df, pd.DataFrame)
        #     best_idx = df['mean_validation_reward'].argmax()
        #     best_step = int(df['num_timesteps'][best_idx])
        #     best_ckpt = os.path.join(path, "model_step_{}.ckpt".format(best_step))

        #     return best_ckpt
        # raise ValueError

    def log_result(self, verification_results=[], test_result={}, verbose=False):
        if verification_results:
            for i in range(len(self.properties)):
                util.log("%02d" % i, level=CONSTANT.WARNING, style=CONSTANT.STYLE_YELLOW, end=" ")
                if verbose:
                    util.log(self.properties[i], level=CONSTANT.WARNING)
                    util.log(self.verification_results[i], level=CONSTANT.WARNING,
                             style=CONSTANT.STYLE_GREEN if self.verification_results[i][0] == True else CONSTANT.STYLE_RED)
                else:
                    util.log(verification_results[i][0], level=CONSTANT.WARNING,
                             style=CONSTANT.STYLE_GREEN if verification_results[i][0] == True else CONSTANT.STYLE_RED)

        if test_result:
            util.log(test_result, level=CONSTANT.INFO)

    def exec_constraint(self, code, x, y):
        exec(code)
        return locals()[drlp.IS_VIOLATED_ID](x, y)

    def RewardAPI(self, x, y, rwd: float):
        # TODO
        pass
        # for property in self.properties:
        #     x = [x]
        #     y = [y]
        #     code = self.get_constraint(property)
        #     violated = self.exec_constraint(code, x, y)
        #     rwd = self.reward(violated, rwd)
        # return rwd

    def get_constraint(self, property):   # TODO
        constraint, dynamics, statics = drlp.parse_drlp_get_constraint(property)
        util.log("## Constraint:\n", constraint.obj)
        util.log(constraint.obj)
        code = drlp.parse_constaint_to_code(constraint,dynamics,statics)
        return code, dynamics, statics


def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# %%
