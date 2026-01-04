"""
稳定性检测模块
用于判断 throughput 和 latency 是否收敛
"""


class StabilityDetector:
    def __init__(self, window_size=10, check_interval=5, epsilon=1e-3, min_steps=100):
        """
        初始化稳定性检测器

        参数:
            window_size: 滑动窗口大小
            check_interval: 检查间隔（每多少步检查一次）
            epsilon: 稳定阈值（相邻记录差值 < epsilon 视为稳定）
            min_steps: 最小运行步数（防止早熟收敛）
        """
        self.window_size = window_size
        self.check_interval = check_interval
        self.epsilon = epsilon
        self.min_steps = min_steps

        self.throughput_history = []
        self.latency_history = []
        self.step_count = 0

    def update(self, throughput, latency):
        """每步更新数据"""
        self.throughput_history.append(throughput)
        self.latency_history.append(latency)
        self.step_count += 1

    def should_check(self):
        """是否应该检查稳定性"""
        return self.step_count % self.check_interval == 0

    def is_stable(self):
        """检查是否稳定（throughput 和 latency 都稳定）"""
        if self.step_count < self.min_steps:
            return False

        if len(self.throughput_history) < self.window_size + 1:
            return False

        # 检查 throughput 稳定性
        if not self._is_series_stable(self.throughput_history):
            return False

        # 检查 latency 稳定性
        if not self._is_series_stable(self.latency_history):
            return False

        return True

    def _is_series_stable(self, series):
        """检查序列是否稳定（相邻差值都 < epsilon）"""
        recent = series[-self.window_size:]
        for i in range(1, len(recent)):
            if abs(recent[i] - recent[i - 1]) >= self.epsilon:
                return False
        return True

    def get_stable_values(self):
        """获取稳定后的值（取最近窗口的平均值）"""
        if not self.is_stable():
            return None, None

        throughput = sum(self.throughput_history[-self.window_size:]) / self.window_size
        latency = sum(self.latency_history[-self.window_size:]) / self.window_size
        return throughput, latency

    def get_current_values(self):
        """获取当前最新值"""
        if self.throughput_history and self.latency_history:
            return self.throughput_history[-1], self.latency_history[-1]
        return None, None

    def reset(self):
        """重置检测器"""
        self.throughput_history = []
        self.latency_history = []
        self.step_count = 0
