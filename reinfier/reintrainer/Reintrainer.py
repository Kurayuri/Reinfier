import re
import os
import json
import copy
import subprocess
import numpy as np
import enum
from collections import namedtuple
from typing import Callable, Dict, List, Set, Union, Tuple, Iterable, Sequence
from ..import nn
from ..import algo
from ..import util
from ..import drlp
from ..import Protocal
from ..import Setting
from ..import CONST
from ..import interpretor
from ..import interface
from ..util.TimerGroup import TimerGroup
from ..common.Feature import Dynamic, Static, Feature, Interval
from ..common.DRLP import DRLP
from ..common.NN import NN
from ..common.type_aliases import PropertyFeatures, WhichFtr
# from bayes_opt import BayesianOptimization
INPUT_EPS_MIN = 0.0
INPUT_EPS_MAX = 0.2
INPUT_EPS_PRECISION = 1e-2


def choose_curriculum():
    pass


PropertyFeatures = namedtuple('PropertyFeatures', ['input', 'output'])


class ReintrainerFlag(enum.Flag):
    OFF = 0
    GEN = enum.auto()
    VIOL = enum.auto()
    DIST = enum.auto()
    DENS = enum.auto()
    GAP = enum.auto()
    TRAC = enum.auto()
    ALL = ~0


Flag = ReintrainerFlag


class Reintrainer:
    '''
    Reintrainer: Property Training Framework for Reinforcement Learning
    '''
    Flag = ReintrainerFlag

    GET_REWARD_FUNC_PARA_REWARD_ID = "reward"

    REWARD_API_FILENAME = "reward_api.py"

    HP_NORM_P1 = "NORM_P1"
    HP_NORM_P2 = "NORM_P2"
    HP_ALPHA = "ALPHA"
    HP_REWARD_CONST = "PD"
    HP_REWARD_TRACE_GAMME = "REWARD_TRACE_GAMME"
    HP_REWARD_TRACE_DEPTH = "REWARD_TRACE_DEPTH"

    def __init__(self, properties: Sequence[DRLP],
                 train_api: Callable | str,
                 save_dirpath: str,
                 verifier: str,
                 init_model_dirpath: str = None,
                 round_exsited: int = -1,
                 reward_api_type: type = str,
                 test_api: Callable | str | None = None,
                 onnx_filename: str = "model.onnx",
                 test_log_filename: str = "test.log",
                 verify_log_filename: str = "verify.log",
                 time_log_filename: str = "time.log",
                 hyperparameters: dict = {},
                 curriculum_api: Callable = None,
                 flag: Flag = Flag.ALL,
                 gap_features: Dict[int, Tuple[str, int]] = {},
                 observation_space: Sequence[Tuple[int, int]] = []
                 ):
        # Proeprty
        self.properties = properties
        self.properties_apply = []
        self.verifier = verifier

        self.flag = flag

        # Init gap_feature
        self.gap_features = {}
        for idx, property in enumerate(self.properties):
            if idx in gap_features:
                self.gap_features[idx] = gap_features[idx]
            else:
                self.gap_features[idx] = (Protocal.DNNP.Input.Id, 0)

        self.observation_space = [Interval(*x) for x in observation_space]

        # Hyperparameter
        self.HPS = {self.HP_NORM_P1: 1,
                    self.HP_NORM_P2: 2,
                    self.HP_ALPHA: 20,
                    self.HP_REWARD_TRACE_DEPTH: 20,
                    self.HP_REWARD_TRACE_GAMME: 0.9
                    }

        self.HPS.update(hyperparameters)

        # External API
        self.curriculum_api = curriculum_api
        self.train_api = train_api
        self.test_api = test_api

        # Internal API
        self.reward_api = None
        if reward_api_type:
            if reward_api_type == callable:
                self.reward_api = self.maker_RewardAPI()
            elif reward_api_type == str:
                self.reward_api = self.REWARD_API_FILENAME
        if self.reward_api is None:
            # TODO
            self.flag = Flag.OFF

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
                        f"Detected save_path {self.save_dirpath} already exists, but Reintrainer does not start in resuming training mode.", itype=CONST.INTERACTIVE_ITYPE_y_or_N):
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
        self.loggers = []
        if self.test_api:
            self.logger_test = open(os.path.join(
                self.save_dirpath, self.test_log_filename), self.global_log_mode)
            self.loggers.append(self.logger_test)
        self.logger_verify = open(os.path.join(
            self.save_dirpath, self.verify_log_filename), self.global_log_mode)
        self.loggers.append(self.logger_verify)

        # Timer
        self.timerGroup = TimerGroup(
            ["Generation", "Training", "Verification", "Testing"])

        # DYN
        self.properties_dynamics = [None for i in range(len(self.properties))]
        self.properties_statics = [None for i in range(len(self.properties))]

        # self.model_select = 'latest'
    def __del__(self):
        try:
            for logger in self.loggers:
                logger.close()
        except:
            pass

    def train(self, round: int, cycle: int):

        # %% Init
        util.log_prompt(4, "Init", style=CONST.STYLE_CYAN)
        util.log("Model: ", self.curr_model_path, level=CONST.INFO)

        logLevel = Setting.set_LogLevel(CONST.ERROR)
        init_verification_results = self.call_verify_api()
        init_test_result = self.call_test_api()
        Setting.set_LogLevel(logLevel)
        util.log_prompt(3, "Summary Part", style=CONST.STYLE_YELLOW)
        util.log(self.curr_model_dirpath, level=CONST.INFO)
        self.log_result(init_verification_results, init_test_result)

        # %% Start
        self.timerGroup.reset()
        self.timerGroup.start()
        for self.round in range(self.round + 1, self.round + 1 + round):
            if Flag.GAP in self.Flag:
                self.HPS[self.HP_ALPHA] /= 2
            util.log_prompt(4, "Round %03d" % self.round, style=CONST.STYLE_CYAN)

            self.next_model_dirpath = self.make_next_model_dir(self.round)

            # %% Train
            util.log_prompt(3, "Generation Part", style=CONST.STYLE_BLUE)

            # % Generation
            self._generate_constraint()
            self.generate_reward()

            self.timerGroup.switch()
            # % Train
            util.log_prompt(3, "Training Part", style=CONST.STYLE_BLUE)
            self.call_train_api(total_cycle=cycle)

            self.curr_model_dirpath = self.next_model_dirpath
            self.curr_model_path = os.path.join(self.curr_model_dirpath, self.onnx_filename)
            util.log("## Current model dirpath: \n%s" % self.curr_model_dirpath, level=CONST.INFO)

            self.timerGroup.switch()
            # %% Verify
            util.log_prompt(3, "Verification Part", style=CONST.STYLE_MAGENTA)
            util.log("Try to verify model:", self.curr_model_path, level=CONST.INFO)

            self.verification_results = self.call_verify_api()

            util.log("\n## Verification Results: ", level=CONST.WARNING)
            self.log_result(self.verification_results, verbose=True)

            self.timerGroup.switch()
            # %% Test
            util.log_prompt(3, "Testing Part", style=CONST.STYLE_GREEN)
            test_result = self.call_test_api()

            self.timerGroup.switch()
            # %% Time
            util.log_prompt(3, "Summary Part", style=CONST.STYLE_YELLOW)
            self.timerGroup.dump(os.path.join(self.save_dirpath, "time.log"))

            times = self.timerGroup.get().values()

            util.log("Round %03d:" % self.round, level=CONST.INFO)
            util.log(self.curr_model_dirpath, level=CONST.INFO)
            self.log_result(self.verification_results, test_result)
            util.log("Reward Generation Time: %.2f s    Training Time: %.2f s    Verification Time: %.2f s    Test Time: %.2f s" % (
                tuple(times)), level=CONST.INFO)
            util.log("Round Time: %.2f s    Total Time: %.2f s" %
                     (sum(times), self.timerGroup.now()[2]), level=CONST.INFO)

        # %% Final
        self.timerGroup.stop()
        self.timerGroup.dump(os.path.join(self.save_dirpath, "time.log"))

        util.log_prompt(4, "Final", style=CONST.STYLE_CYAN)
        util.log("Model: ", self.curr_model_path, level=CONST.INFO)
        util.log("Total Time: %.2f s" %
                 (self.timerGroup.now()[2]), level=CONST.INFO)

        util.log("From:", level=CONST.INFO, style=CONST.STYLE_RED)
        self.log_result(init_verification_results, init_test_result)

        util.log("\nTo:", level=CONST.INFO, style=CONST.STYLE_GREEN)
        self.log_result(self.verification_results, test_result)

    def _generate_constraint(self):
        if Flag.GEN not in self.flag:
            return

        for i in range(len(self.properties)):
            code, dynamics, statics = self.get_constraint(self.properties[i])

            self.properties_dynamics[i] = dynamics
            self.properties_statics[i] = statics

        with open(os.path.join(self.next_model_dirpath, self.REWARD_API_FILENAME), "w") as f:
            f.write(code)
        exec(code)
        self.Is_Violated_Func_Func = locals()[Protocal.API.Reward.IsViolated.Id]

        return code

    def generate_reward(self, to_append: bool = True):
        if Flag.GEN not in self.flag:
            return

        mode = "a+" if to_append else "w"

        code_get_reward = ""

        # self.update_features()

        def autogen():
            # TODO
            dynamics: Dict[int, Dynamic]
            statics: Dict[int, Static]
            dynamics = self.properties_dynamics[0].input
            statics = self.properties_statics[0].input

            # # TODO for Aurora
            # for i in range(10):
            #     idx = i * 3 + 0
            #     dk = dynamics[idx]
            #     statics[idx] = Static(dk.lower, dk.upper,
            #                           dk.lower_closed, dk.upper_closed)
            #     dynamics.pop(idx)

            #     idx = i * 3 + 1
            #     dk = dynamics[idx]
            #     statics[idx] = Static(dk.lower, dk.upper,
            #                           dk.lower_closed, dk.upper_closed)
            #     dynamics.pop(idx)
            # ###########

            # Not first round and rho
            inputFtrs = {**dynamics, **statics}
            if self.curr_model_dirpath and (Flag.DENS in self.flag):
                network = NN(self.curr_model_path)
                for idx, dynamic in dynamics.items():
                    dynamic.lower_rho = self.measure_rho(network, inputFtrs, idx, "Lower")
                    dynamic.upper_rho = self.measure_rho(network, inputFtrs, idx, "Upper")
                    dynamic.weight = (dynamic.lower_rho + dynamic.upper_rho) / 2

            util.log("Importance Weight:", {idx: dynamic.weight for idx, dynamic in dynamics.items(
            )}, level=CONST.WARNING, style=CONST.STYLE_RED)

            dist_srcs = []

            for idx, dynamic in dynamics.items():
                mid = (dynamic.lower_rho * dynamic.lower + dynamic.upper_rho *
                       dynamic.upper) / (dynamic.lower_rho + dynamic.upper_rho)

                # # Edge condition
                # if len(inputFtrs) == len(self.observation_space):
                #     if dynamic.lower == self.observation_space[idx].lower:
                #         mid = dynamic.lower
                #     if dynamic.upper == self.observation_space[idx].upper:
                #         mid = dynamic.upper
                #     if dynamic.lower == self.observation_space[idx].lower and dynamic.upper == self.observation_space[idx].upper:
                #         continue

                src = f"dists_x[{idx}] = dist(x[0][{idx}], {dynamic.lower}, {dynamic.upper}, {mid}, {dynamic.lower_rho}, {dynamic.upper_rho})"
                dist_srcs.append(src)

            # TODO
            # dist_srcs.pop(0)

            code_get_reward = f'''
def {Protocal.API.Reward.GetReward.Id}({Protocal.API.Reward.GetReward.Param.Observation.Id}, {Protocal.API.Reward.GetReward.Param.Action.Id}, {Protocal.API.Reward.GetReward.Param.Reward.Id}, {Protocal.API.Reward.GetReward.Param.Violated.Id}):
    p1 = {self.HPS[self.HP_NORM_P1]}
    p2 = {self.HPS[self.HP_NORM_P2]}
    alpha = {self.HPS[self.HP_ALPHA]}
        
    def dist(val, lower, upper, mid, lower_rho, upper_rho):
        if val > upper or val < lower:
            return 0, 0

        if val > mid:
            return {'((upper-val)/(upper-mid))**p1' if Flag.DENS in self.flag else 'upper-val'}, upper_rho
        else:
            return {'((val-lower)/(mid-lower))**p1' if Flag.DENS in self.flag  else 'val-lower'}, lower_rho

    if {Protocal.API.Reward.GetReward.Param.Violated.Id}:
        dists_x = dict()
''' + "".join(["        " + srs + "\n" for srs in dist_srcs]) + f'''
        sum_1 = sum([dist[1]*(dist[0]**p2) for dist in dists_x.values()])
        sum_2 = sum([dist[1] for dist in dists_x.values()])
        Dist_x = (sum_1)**(1/p2)/sum_2
        Fs = - 1 * Dist_x
        reward = reward + {'Fs * alpha' if self.HP_REWARD_CONST not in self.HPS else self.HPS[self.HP_REWARD_CONST]}
    else:
        reward = reward
    return reward
''' if Flag.DIST in self.flag else Protocal.API.Reward.GetReward.Template
            return code_get_reward

#         def nogen():
#             get_reward_code = f'''
# def {self.Protocal.API.Reward.IsViolated.Id}(x, y):
#     if np.all([[1.5,1.5]]>x) and np.all([[-1.5,-1.5]]<x):
#         return True, True
#     else:
#         return True, False

# def {self.Protocal.API.REWARD.GET_REWARD.ID}(x, y, reward, violated):
#     from math import sqrt
#     # return reward - 2*sqrt((min(abs(x[0,0]-0.2),abs(x[0,0]-0))/5)**2 + (min(abs(x[0,1] - 0.05),abs(x[0,1] - 0.3)))**2)
#     # return reward - 0.5*(min(abs(x[0,0]-0.2),abs(x[0,0]-0))+(min(abs(x[0,1] - 0.05),abs(x[0,1] - 0.3)))) b1_p4
#     # return reward - 4/(min(abs(x[0,0]-1),abs(x[0,0]--1)) + min(abs(x[0,1] - 1),abs(x[0,1] - -1)) + min(abs(x[0,1] -1.57),abs(x[0,1] --1.57)) + min(abs(x[0,1] -1),abs(x[0,1] - -1)))
#     return reward  - {self.HPS[self.HP_PD]}/((min(abs(x[0,0]-1.5), abs(x[0,0]+1.50)) + (min(abs(x[0,1]-1.5), abs(x[0,1] + 1.5))))+1) # b1_2
#     # return reward - sqrt(min(abs(x[0,0]-0.2),abs(x[0,0]-0))**2 + min(abs(x[0,1]-0.05),abs(x[0,1]-0.3))**2)

#     # return reward
# '''
            # return code_get_reward

        code_get_reward = autogen()

        util.log(code_get_reward)
        with open(os.path.join(self.next_model_dirpath, self.REWARD_API_FILENAME), mode) as f:
            f.write(code_get_reward)

        exec(code_get_reward)
        self.Get_Reward_Func = locals()[Protocal.API.Reward.GetReward.Id]

        return code_get_reward

    def measure_rho(self, network: NN, inputFtrs: Dict[int, Feature], index: int, lower_or_upper: str):
        util.log("Dynamic", index, level=CONST.INFO)

        is_lower = lower_or_upper[0].lower() == "l"

        input_size, output_size = network.size()

        x_base = [[0 for i in range(input_size)]]
        for idx, ftr in inputFtrs.items():
            x_base[0][idx] = ftr.lower if is_lower else ftr.upper

        def verification():
            logLevel = Setting.set_LogLevel(CONST.CRITICAL)
            y_base = nn.run_onnx(network, np.array(
                x_base, dtype=np.float32))[0]

            x_eps_id = "x_eps"
            kwargs = {
                x_eps_id: {"lower_bound": INPUT_EPS_MIN,
                           "upper_bound": INPUT_EPS_MAX,
                           "precise": INPUT_EPS_PRECISION,
                           "method": "binary", },
            }

            values = {
                "y_eps": 1e-0,
                "y_base": float(y_base[0][0])
            }

            bound = [str(x_base[0][i]) for i in range(input_size)]

            if is_lower == 0:
                bound_lower = ",".join(bound)
                bound[index] = bound[index] + "+" + x_eps_id
                bound_upper = ",".join(bound)
            else:
                bound_upper = ",".join(bound)
                bound[index] = bound[index] + "-" + x_eps_id
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
            bps = algo.search_breakpoints(
                network, property, kwargs, 0.01, self.verifier, k_max=1, to_induct=False)
            inline_bps, inline_bls = interpretor.analyze_break_points(bps)
            answer, __, __ = interpretor.answer_importance_analysis(inline_bps)
            Setting.set_LogLevel(logLevel)

            return answer[0] if answer is not None else 1

        def grad():

            if is_lower:
                lower = x_base[0][index]
                upper = x_base[0][index] + INPUT_EPS_MIN
            else:
                lower = x_base[0][index] - INPUT_EPS_MIN
                upper = x_base[0][index]

            (max_sum, max_val), (min_sum, min_val) = interpretor.measure_sensitivity(network=network, input=x_base, index=(0, index),
                                                                                     lower=lower, upper=upper, precision=INPUT_EPS_PRECISION)
            return max(abs(max_sum), abs(min_sum))
        return grad()

    def call_train_api(self, **kwargs):
        util.log("Training...", level=CONST.INFO)

        if isinstance(self.train_api, Callable):
            kwargs["next_model_dirpath"] = self.next_model_dirpath
            kwargs["total_cycle"] = kwargs['total_cycle']

            if self.reward_api:
                kwargs["reward_api"] = self.reward_api
            if self.curr_model_dirpath:
                kwargs["curr_model_dirpath"] = self.curr_model_dirpath
            if Flag.TRAC in self.flag:
                kwargs["enabled_trace"] = True
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
                f"--reward_api         {self.reward_api} " if self.reward_api else "" \
                f"--curr_model_dirpath {self.curr_model_dirpath}" if self.curr_model_dirpath else "" \
                f"--enabled_trace      True" if Flag.TRAC in self.flag else "" \
                ""
            cmd += cmd_suffix

            util.log(cmd, level=CONST.INFO)
            proc = subprocess.run(cmd, shell=True, capture_output=False)

    def call_test_api(self, **kwargs):
        if self.test_api is None:
            return None
        if self.curr_model_dirpath is None:
            return None

        util.log("Testing...", level=CONST.INFO)
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

            util.log(cmd, level=CONST.INFO)
            proc = subprocess.run(cmd, shell=True, capture_output=False)

            result = self.load_test_result(os.path.join(
                self.curr_model_dirpath, self.test_log_filename))

        self.logger_test.write(
            f"Round {self.round:03d} Test: mean_reward:{result['mean_reward']:.2f} +/- {result['std_reward']:.2f}\n")
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
                ans = algo.verify(network, property, verifier=self.verifier,
                                 to_induct=False, k_max=1, reachability=False)
                verification_results.append(ans)

            anss = {'%02d' % i: verification_results[i][0]
                    for i in range(len(self.properties))}
            self.logger_verify.write(
                f"Round {self.round:03d} Verify: {anss}\n")
            self.logger_verify.flush()
        return verification_results

    def make_next_model_dir(self, round: int):
        path = os.path.join(self.save_dirpath, f"round_%03d" % (round))
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            util.log(e, level=CONST.ERROR)
        return path

    def load_test_result(self, path: str) -> Dict:
        result = None
        try:
            f = open(path, "r")
            try:
                result = json.load(f)
            except Exception:
                util.log(
                    f"Invalid test result file at {path}.", level=CONST.WARNING, style=CONST.STYLE_YELLOW)
            f.close()
        except Exception:
            util.log(f"Could not open test result file at {path}.",
                     level=CONST.WARNING, style=CONST.STYLE_YELLOW)

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
                util.log("%02d" % i, level=CONST.WARNING,
                         style=CONST.STYLE_YELLOW, end=" ")
                if verbose:
                    util.log(self.properties[i], level=CONST.WARNING)
                    util.log(self.verification_results[i], level=CONST.WARNING,
                             style=CONST.STYLE_GREEN if self.verification_results[i][0] == True else CONST.STYLE_RED)
                else:
                    util.log(verification_results[i][0], level=CONST.WARNING,
                             style=CONST.STYLE_GREEN if verification_results[i][0] == True else CONST.STYLE_RED)

        if test_result:
            util.log(test_result, level=CONST.INFO)

    def maker_RewardAPI(self):
        def _maker_Is_Violated_Func(x, y):
            return self.Is_Violated_Func(x, y)

        def _maker_Get_Reward_Func(x, y, reward, violated):
            return self.Get_Reward_Func(x, y, reward, violated)

        return {
            Protocal.API.Reward.IsViolated.Id: _maker_Is_Violated_Func,
            Protocal.API.Reward.GetReward.Id: _maker_Get_Reward_Func
        }

    def Get_Reward_Func(self, x, y, reward, violated):
        return reward

    def Is_Violated_Func(self, x, y):
        return False, False

    def get_constraint(self, property):   # TODO
        constraint, dynamics, statics = drlp.parse_drlp_get_constraint(property)
        util.log("## Constraint:")
        util.log(constraint.obj)

        code = ""
        if Flag.GEN in self.flag:
            code = Protocal.API.Reward.IsViolated.Template
        if Flag.VIOL in self.flag:
            code = drlp.parse_constaint_to_code(constraint, dynamics, statics)
        util.log("## Constraint Code:")
        util.log(code)
        return code, dynamics, statics

    def module_call(self, module):
        module_name, module_level, module_call = module
        util.log_prompt(
            module_level, f"{module_name} Part", style=CONST.STYLE_BLUE)

    def update_features(self):
        for idx, property in enumerate(self.properties):
            gap_feature = self.gap_features[idx]
            tmp = self.properties_statics[idx]
            self.properties_statics[idx] = self.properties_dynamics[idx]
            self.properties_dynamics[idx] = tmp

            if gap_feature[0] == Protocal.DRLP.Input.Id:
                feature = self.properties_statics[idx].input.pop(gap_feature[1])
                dynamic = PropertyFeatures({gap_feature[1]: feature}, {})

            else:  # gap_feature[0] == Protocal.DRLP.Output.Id
                feature = self.properties_statics[idx].output.pop(gap_feature[1])
                dynamic = PropertyFeatures({}, {gap_feature[1]: feature})

            self.properties_dynamics[idx] = dynamic

    def measure_gap(self, inputFtrs: Dict[int, Feature], outputFtrs: Dict[int, Feature], whichFtr: WhichFtr):
        gapFtr: Feature
        gapFtr = inputFtrs[whichFtr.index] if whichFtr.io == Protocal.DRLP.Input.Id else outputFtrs[whichFtr.index]
        _gapFtr = copy.deepcopy(gapFtr)

        def be_slack(gapFtr: Feature, io: str):
            if io == Protocal.DRLP.Input.Id:
                gapFtr.lower += 1
                gapFtr.upper -= 1
            else:
                gapFtr.lower -= 1
                gapFtr.upper += 1
