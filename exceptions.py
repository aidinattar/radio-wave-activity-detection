class OptionIsFalseError(Exception):
    def __init__(self, option_name):
        self.option_name = option_name

    def __str__(self):
        return f"The '{self.option_name}' option is set to False. The code cannot be run."

class WorkToDoError(Exception):
    def __init__(self, work_name):
        self.work_name = work_name

    def __str__(self):
        return f"The '{self.work_name}' work is not done. The code cannot be run."