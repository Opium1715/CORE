import os


class P_MRR_Record:
    def __init__(self, save_path, frequency):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.frequency = frequency

    def __call__(self, precision, mrr, *args, **kwargs):
        with open(self.save_path + '/precision.txt', 'a') as f:
            f.write(str(precision) + '\n')
        with open(self.save_path + '/mrr.txt', 'a') as f:
            f.write(str(mrr) + '\n')
