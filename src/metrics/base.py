class MetricBase:
    def evaluate(self, predict, target, **kwargs):
        raise NotImplementedError

    def accumulate(self, *args, **kwargs):
        raise NotImplementedError

    def results(self):
        raise NotImplementedError

    def __call__(self, predict, target, **kwargs):
        result = self.evaluate(predict=predict, target=target, **kwargs)
        self.accumulate(result)
        return result
