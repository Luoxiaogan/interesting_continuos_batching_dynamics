"""
LLM推理调度问题模拟器 - Even simpler setting
支持确定性到达和随机到达（Gamma分布）
"""

import numpy as np

class LLMSchedulerSimulator:
    def __init__(self, l0, l1, B, X0, Qe0, lambda_rate, b0, b1, if_float=False,
                 admission_threshold=0, admission_upper_bound=None, verbose=False,
                 seed=None, stochastic=True):
        """
        初始化模拟器

        参数:
        l0: 作业初始大小的基础部分
        l1: 作业处理阶段数
        B: GPU容量限制
        X0: 初始状态X(0)，长度为l1的列表
        Qe0: 初始外部队列长度
        lambda_rate: 到达率（均值参数）
        b0, b1: 批处理时间参数 s(n) = b0 + b1*Z(n)
        admission_threshold: admission control阈值s（小s），只有当Qe+Qr>=s时才做admission
        admission_upper_bound: admission上界S（大S），每次admission数量不超过S，None表示无上界
        verbose: 是否打印详细日志
        seed: 随机种子，用于复现实验（None 表示不固定）
        stochastic: 是否使用随机到达（Gamma分布），False 则使用确定性到达
        """
        self.l0 = l0
        self.l1 = l1
        self.B = B
        self.X = X0.copy()  # 当前状态，X[i]表示处理阶段i的作业数
        self.Qe = Qe0  # 外部到达队列
        self.Qr = 0  # 重新排队的作业队列
        self.lambda_rate = lambda_rate
        self.b0 = b0
        self.b1 = b1
        self.T = 0
        self.total_completion = 0
        self.if_float = if_float
        self.X0 = X0.copy()
        self.admission_threshold = admission_threshold  # 小s
        self.admission_upper_bound = admission_upper_bound  # 大S
        self.verbose = verbose
        self.stochastic = stochastic

        # 随机数生成器（用于随机到达）
        self.rng = np.random.default_rng(seed)

        # Latency 跟踪相关变量
        self.total_latency_token_product = 0  # sum(latency * tokens)
        self.total_completed_tokens = 0        # 完成的总 token 数
        self.admission_records = {}            # {batch_id: {admission_time, token_count, jobs_remaining}}
        self.current_admission_batch_id = 0

        # 为初始 X0 的 job 创建虚拟 admission_record
        # X0[i] 的 job 在 stage i，它们将在 (l1 - i) 个 batch 后完成
        # 假设它们在 T = -(l1 - i) * estimated_service_time 时被 admit
        # 估算初始 service time
        initial_Z = sum((l0 + i + 1) * X0[i] for i in range(l1))
        estimated_service_time = b0 + b1 * initial_Z

        for i in range(l1):
            if X0[i] > 0:
                # stage i 的 job 将在 (l1 - 1 - i) 个 batch 后完成
                # 它们应该在 T = -((l1 - 1 - i) + 1) * service_time 时被 admit
                # 即在 (l1 - i) 个 batch 之前
                batches_until_completion = l1 - i
                virtual_admission_time = -batches_until_completion * estimated_service_time
                self.current_admission_batch_id -= 1  # 使用负数 batch_id
                self.admission_records[self.current_admission_batch_id] = {
                    'admission_time': virtual_admission_time,
                    'jobs_remaining': X0[i]
                }

        # 重置 batch_id 计数器（虚拟的用负数，真实的从 1 开始）
        self.current_admission_batch_id = 0

        # 记录每次 admission 的数目
        self.admission_history = []

        # 记录每个 batch 的时序数据
        self.batch_history = []  # [{batch_idx, T, throughput, latency, cumulative_eviction}, ...]
        self.cumulative_eviction = 0

        # 记录历史
        self.history = []
        self.n = 0  # 当前批次
        if self.verbose:
            print("*"*50)
            print(f"INIT:n={self.n}, X={self.X}, Qe={self.Qe:.3f}, Qr={self.Qr}, B={self.B}, lambda={self.lambda_rate}")
            print(f"Virtual admission records for X0: {self.admission_records}")
            print("*"*50)
        
    def compute_batch_size(self):
        """计算当前批次大小Z(n)，使用 active memory (l0+i+1)"""
        Z = 0
        for i in range(0, self.l1):
            Z += (self.l0 + i + 1) * self.X[i]
        return Z
    
    def compute_service_time(self, Z):
        """计算服务时间s(n)"""
        return self.b0 + self.b1 * Z
    
    def update(self):
        """执行一次更新（处理一个批次）"""

        if sum(self.X[i] for i in range(0, self.l1))==0 and (self.Qe+self.Qr < 1):
            raise Exception("队列数目小于0, X空了, batch驱动无法解!")

        if self.verbose:
            print("\n"+"-"*50)

        # 1. 首先做 admission / eviction 
        # 从高 stage 向下填充 B 和 M (M=B)
        token_in_need = 0
        X_prime = [0] * self.l1
        for i in range(self.l1-1, -1, -1):
            # 从 l1-1 到 0
            if token_in_need + (self.l0 + i + 1) * self.X[i] <= self.B:
                X_prime[i] = self.X[i]
                token_in_need += (self.l0 + i + 1) * self.X[i]
            else:
                available_tokens = self.B - token_in_need
                length = self.l0 + i + 1
                if self.if_float:
                    number = available_tokens / length
                else:
                    number = int(available_tokens / length)
                assert (number <= self.X[i])
                X_prime[i] = number
                break
        the_same = True
        eviction = 0
        for i in range(0, self.l1):
            assert (X_prime[i] <= self.X[i])
            if X_prime[i] != self.X[i]:
                the_same = False
                eviction += self.X[i] - X_prime[i]
        self.cumulative_eviction += eviction  # 累计 eviction

        # 记录 admission 之前的 batch token 数 (active memory)
        tokens_before_admission = sum((self.l0 + i + 1) * X_prime[i] for i in range(self.l1))

        if the_same:
            # 说明还可能做 admission
            # admission control: 只有当 Qe + Qr >= admission_threshold (小s) 时才做 admission
            if self.Qe + self.Qr >= self.admission_threshold:
                available_tokens = self.B - token_in_need
                length = self.l0 + 1
                if self.if_float:
                    number = available_tokens / length
                    admission = min(self.Qe + self.Qr, number)
                    # 应用大S上界
                    if self.admission_upper_bound is not None:
                        admission = min(admission, self.admission_upper_bound)
                else:
                    number = int(available_tokens / length)
                    admission = min(int(self.Qe + self.Qr), number)
                    # 应用大S上界
                    if self.admission_upper_bound is not None:
                        admission = min(admission, int(self.admission_upper_bound))
                X_prime[0] += admission
            else:
                # Qe + Qr < admission_threshold (小s), 不做 admission
                admission = 0
            assert (eviction == 0)
        else:
            admission = 0
            assert (eviction >= 0)


        # 2. 执行 batch
        # Z 是 admission 之后的 batch token 数 (active memory)
        Z = sum((self.l0 + i + 1) * X_prime[i] for i in range(self.l1))
        s = self.compute_service_time(Z)
        if self.verbose:
            print(f"n={self.n}, T={self.T:.3f}: after AE, X={X_prime}, sum_X = {sum(X_prime[i] for i in range(0,self.l1))}, Qe={self.Qe:.3f}, Qr={self.Qr}, admission={admission}, eviction={eviction}, Z={Z}, s={s:.3f}")

        # 记录 admission（用于 latency 跟踪）
        self.admission_history.append(admission)  # 记录每次 admission 数目
        if admission > 0:
            self.current_admission_batch_id += 1
            self.admission_records[self.current_admission_batch_id] = {
                'admission_time': self.T,  # 当前时间（在服务时间之前）
                'jobs_remaining': admission
            }

        self.T += s
        if self.verbose:
            print(f"X=X0+{[x - y for x, y in zip(X_prime, self.X0)]}")

        # 首先统一 update admission or eviction
        Qe_prime = self.Qe
        Qr_prime = self.Qr
        if eviction > 0:
            assert (admission == 0)
            Qr_prime += eviction
        else:
            assert(admission <= self.Qe + self.Qr)
            r = min(admission, self.Qr)
            Qr_prime -= r
            assert (Qr_prime >= 0)
            e = min(admission - r, self.Qe)
            Qe_prime -= e
            assert (Qe_prime >= 0)

        # 然后 update arrival
        # 随机到达：Gamma(shape=λs, scale=1)，均值=λs，方差=λs（与泊松相同）
        # 确定性到达：直接用 λs
        if self.stochastic:
            mean_arrivals = self.lambda_rate * s
            if mean_arrivals > 0:
                arrivals = self.rng.gamma(shape=mean_arrivals, scale=1.0)
            else:
                arrivals = 0.0
        else:
            arrivals = self.lambda_rate * s
        Qe_prime = Qe_prime + arrivals

        # 然后执行 batch
        # 检查 active memory 不超过 B
        assert (Z <= self.B)
        completion = X_prime[self.l1-1]
        self.total_completion += completion

        # 处理 completion，计算 latency
        if completion > 0:
            self._process_completions(completion, self.T)

        X_prime_tmp = [0] * self.l1
        for i in range(1, self.l1):
            X_prime_tmp[i] = X_prime[i-1]

        self.X = X_prime_tmp
        self.Qe = Qe_prime
        self.Qr = Qr_prime
        self.n += 1

        # 记录时序数据
        self.batch_history.append({
            'batch_idx': self.n,
            'T': self.T,
            'throughput': self.get_current_throughput(),
            'latency': self.get_current_avg_latency(),
            'cumulative_eviction': self.cumulative_eviction,
            'admission': admission,
            'queue_length': self.Qe + self.Qr,  # 当前 Qe + Qr（admission 后的值）
            'tokens_before_admission': tokens_before_admission,  # admission 前的 batch token 数
            'tokens_after_admission': Z  # admission 后的 batch token 数
        })

        if self.verbose:
            print(f"n={self.n}, T={self.T:.3f}: after B, X={self.X}, sum_X = {sum(self.X[i] for i in range(0,self.l1))}, Qe={self.Qe:.2f}, Qr={self.Qr}, completion={completion}")
            print(f"n={self.n}, T={self.T:.3f}: total_completion={self.total_completion}, throughput={self.total_completion/self.T}, lambda={self.lambda_rate}")

    def _process_completions(self, completion_count, completion_time):
        """处理完成的作业，计算 latency（FIFO 顺序）"""
        remaining = completion_count
        tokens_per_job = self.l0 + self.l1  # 每个作业完成时的总 token 数

        # 按 FIFO 顺序处理 admission batches
        for batch_id in sorted(self.admission_records.keys()):
            if remaining <= 0:
                break

            record = self.admission_records[batch_id]
            jobs_to_complete = min(remaining, record['jobs_remaining'])

            if jobs_to_complete > 0:
                latency = completion_time - record['admission_time']
                tokens_completed = jobs_to_complete * tokens_per_job

                self.total_latency_token_product += latency * tokens_completed
                self.total_completed_tokens += tokens_completed

                record['jobs_remaining'] -= jobs_to_complete
                remaining -= jobs_to_complete

        # 清理已完成的 admission records
        self.admission_records = {
            k: v for k, v in self.admission_records.items()
            if v['jobs_remaining'] > 0
        }

    def get_current_throughput(self):
        """获取当前 throughput（token/sec）"""
        if self.T > 0:
            return self.total_completion * (self.l0 + self.l1) / self.T
        return 0

    def get_current_avg_latency(self):
        """获取当前平均 latency（token-weighted）"""
        if self.total_completed_tokens > 0:
            return self.total_latency_token_product / self.total_completed_tokens
        return 0

    # def record(self, Z, s):
    #     """记录当前状态"""
    #     record = {
    #         'n': self.n,
    #         'Qe': self.Qe,
    #         'Qr': self.Qr,
    #         'X': self.X.copy(),
    #         'Z': Z,
    #         's': s
    #     }
    #     self.history.append(record)
        
    def run(self, steps):
        """运行模拟指定步数"""
        for _ in range(steps):
            self.update()
            
    # def get_history(self):
    #     """获取历史记录"""
    #     return self.history
    
    # def print_state(self):
    #     """打印当前状态"""
    #     print(f"批次 n={self.n}")
    #     print(f"  外部队列 Qe={self.Qe:.2f}")
    #     print(f"  重排队列 Qr={self.Qr:.2f}")
    #     print(f"  GPU状态 X={self.X}")
    #     print(f"  批大小 Z={self.compute_batch_size()}")
    #     print(f"  服务时间 s={self.compute_service_time(self.compute_batch_size()):.2f}")


# 示例使用
if __name__ == "__main__":
    # 设置参数（基于PDF中的例子：l0=1, l1=2）
    l0 = 2
    l1 = 4
    B = 190  # GPU容量
    # a=B/(0.982756*5+0.122845*6+0.1382*7)
    # X0 = [0,  0,  0.982756*a,  0.122845*a,  0.1382*a]  # 初始状态 (B/2, 0), 注意初始状态直接爆炸
    X0 = [0,  20,  11,10]  # 初始状态 (B/2, 0), 注意初始状态直接爆炸
    Qe0 = 1000  # 初始外部队列
    lambda_rate = 12  # 到达率
    b0 = 0.1
    b1 = 0.01
    if_float = True
    
    # 创建模拟器
    sim = LLMSchedulerSimulator(l0, l1, B, X0, Qe0, lambda_rate, b0, b1, if_float)

    t = 100
    
    sim.run(t)
