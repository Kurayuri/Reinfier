class DRLP:
    class Input:
        Id = "x"
        SizeId = "x_size"

    class Output:
        ID = "y"
        SizeID = "y_size"

    class Delimiter:
        Precondition = "@Pre"
        Expectation = "@Exp"

    class Depth:
        Id = "k"


class DNNP:
    class Input:
        Id = "x"


class API:
    class Reward:
        class IsViolated:
            Id = "is_violated"

            class Param:
                class Observation:
                    Id = "x"

                class Action:
                    Id = "y"

            class Return:
                class Occurred:
                    Id = "occurred"

                class Violated:
                    Id = "violated"
            Template = f'''
def {Id}({Param.Observation.Id}, {Param.Action.Id}):
    {Return.Occurred.Id} = True
    {Return.Violated.Id} = True
    return {Return.Occurred.Id},{Return.Violated.Id}
'''

        class GetReward:
            Id = "get_reward"

            class Param:
                class Observation:
                    Id = "x"

                class Action:
                    Id = "y"

                class Reward:
                    Id = "reward"

                class Violated:
                    Id = "violated"

            class Return:
                class Reward:
                    Id = "reward"
            Template = f'''
def {Id}({Param.Observation.Id}, {Param.Action.Id}, {Param.Reward.Id}, {Param.Violated.Id}):
    return {Param.Reward.Id}
'''