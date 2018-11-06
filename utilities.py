import numpy as np

class Report:

    def __init__(self, agent=None):
        if agent is not None:
            self.agent      = agent
        self.plots      = []
        self.averages   = {}
        self.scores     = []
        self.wall_time  = {}

    @staticmethod
    def from_dict(**kwargs):
        import pandas as pd
        report = []
        if kwargs is not None:
            for key, val in kwargs.items():
                report.append((key, val))
        return pd.DataFrame.from_items(report)

    def add_plot(self, data, labels=('Episode', 'Score'), rolling_window=100):
        import matplotlib.pyplot as plt
        import pandas as pd
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if labels is not None:
            x_label, y_label = [label for label in labels]
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        plt.plot([13 for x in range(0, len(data) + 1)], label='Goal')
        plt.plot(np.arange(1, len(data) + 1, step=1), data, label='Episode')
        rolling_mean = pd.Series(data).rolling(rolling_window).mean()
        plt.plot(rolling_mean, label='Rolling 100 Episode Average')
        plt.legend()
        self.plots.append(plt)

    def show_plots(self):
        for plot in self.plots:
            plot.show()

    def run(self, exe, **kwargs):
        self.scores, self.averages = exe(**dict(kwargs, agent=self.agent, log_time=self.wall_time, report=self))

        self.show_plots()
        return self

    def __str__(self):
        from tabulate import tabulate
        import pandas as pd
        import datetime

        report = [('Agent', self.agent)]

        avg_table = pd.DataFrame.from_records([(key, val) for key, val in self.averages.items()])
        report +=  [('Averages', tabulate(avg_table, tablefmt='fancy_grid', headers=["Ep", "Score"], showindex='never'))]

        wall_time_table = pd.DataFrame.from_records([(key, str(datetime.timedelta(seconds=val))) for key, val in self.wall_time.items()] + [('Total', datetime.timedelta(seconds=sum(self.wall_time.values())))])
        report += [('Wall Time', tabulate(wall_time_table, tablefmt='fancy_grid', headers=["Fn", "Time"], showindex='never'))]

        report += [('Episodes', len(self.scores))]

        report_table = pd.DataFrame.from_records(report)
        return tabulate(report_table, tablefmt='fancy_grid', showindex='never')


def timeit(method):
    def timed(*args, **kw):
        import datetime
        from timeit import default_timer as timer
        t_0     = timer()
        result  = method(*args, **kw)
        t_end   = timer()
        seconds = datetime.timedelta(seconds=t_end - t_0)
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            print("\n\nTimeDelta: {}".format(str(t_end - t_0)))
            print("Seconds  : {}".format(str(seconds)))
            kw['log_time'][name] = t_end - t_0
        else:
            print('\n\nExecution Time')
            print('\r{}\t{}'.format(method.__name__, seconds))
        return result

    return timed
