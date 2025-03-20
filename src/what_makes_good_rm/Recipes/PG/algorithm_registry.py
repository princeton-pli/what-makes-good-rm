from what_makes_good_rm.Recipes.PG.custom_grpo_trainer import CustomGRPOTrainer
from what_makes_good_rm.Recipes.PG.custom_ppo_trainer import CustomPPOTrainer
from what_makes_good_rm.Recipes.PG.custom_rloo_trainer import CustomRLOOTrainer

ALGORITHM_TRAINER_CLASSES = {
    'RLOO': CustomRLOOTrainer,
    'PPO': CustomPPOTrainer,
    'GRPO': CustomGRPOTrainer,
}
