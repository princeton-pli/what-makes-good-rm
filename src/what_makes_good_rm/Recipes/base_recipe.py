from what_makes_good_rm.Arguments import CustomTrainingArguments


class BaseRecipe:
    def __init__(self, config=None):
        self.config = config

    def run(self, **kwargs):
        raise NotImplementedError
