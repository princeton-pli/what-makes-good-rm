from what_makes_good_rm.Data.Utils import ArmoRMWrapper, GeneralRMWrapper

REWARD_MODEL_WRAPPER_REGISTRY = {
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": ArmoRMWrapper,
    "GeneralRM": GeneralRMWrapper
}