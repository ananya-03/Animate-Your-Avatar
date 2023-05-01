from brax import envs
from .humanoid_mimic import HumanoidMimic
from .humanoid_mimic_train import HumanoidMimicTrain


def register_mimic_env():
    envs.register_environment('humanoid_mimic', HumanoidMimic)
    envs.register_environment('humanoid_mimic_train', HumanoidMimicTrain)