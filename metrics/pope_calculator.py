from .calculator import calculator
from .pope_metrics import PopeMetricParser
class PopeCalculator(calculator):
    def __init__(self):
        super().__init__(PopeMetricParser())