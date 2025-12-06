"""
LLM推理调度问题模拟器 - Even simpler setting
基于确定性到达率的单类作业模型
"""

import numpy as np

class LLMSchedulerSimulator:
    def __init__(self, l0, l1, B, X0, Qe0, lambda_rate, b0, b1, if_float=False):
        """
        初始化模拟器
        
        参数:
        l0: 作业初始大小的基础部分
        l1: 作业处理阶段数
        B: GPU容量限制
        X0: 初始状态X(0)，长度为l1的列表
        Qe0: 初始外部队列长度
        lambda_rate: 确定性到达率
        b0, b1: 批处理时间参数 s(n) = b0 + b1*Z(n)
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
        
        # 记录历史
        self.history = []
        self.n = 0  # 当前批次
        print("*"*50)
        print(f"INIT:n={self.n}, X={self.X}, Qe={self.Qe:.3f}, Qr={self.Qr}, B={self.B}, lambda={self.lambda_rate}")
        print("*"*50)
        
    def compute_batch_size(self):
        """计算当前批次大小Z(n)"""
        Z = 0
        for i in range(0, self.l1): # 0 \to l1-1, 这里做了修改
            Z += (self.l0 + i) * self.X[i] 
        return Z
    
    def compute_service_time(self, Z):
        """计算服务时间s(n)"""
        return self.b0 + self.b1 * Z
    
    def update(self):
        """执行一次更新（处理一个批次）"""

        if sum(self.X[i] for i in range(0, self.l1))==0 and (self.Qe+self.Qr == 0):
            raise Exception("queue length is zero, and batch is empty.")

        print("\n"+"-"*50)

        # 1. first, admission / eviction 
        # from higher to lower
        token_in_need = 0
        X_prime = [0] * self.l1
        for i in range(self.l1-1, -1, -1):
            # from l1-1 to 0
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
        if the_same:
            # may still need admission
            available_tokens = self.B - token_in_need
            length = self.l0 + 1
            if self.if_float:
                number = available_tokens / length
                admission = min(self.Qe + self.Qr, number)
            else:
                number = int(available_tokens / length)
                admission = min(int(self.Qe + self.Qr), number)
            X_prime[0] += admission
            assert (eviction == 0)
        else:
            admission = 0
            assert (eviction >= 0)


        # 2. execute batch
        Z = sum((self.l0 + i) * X_prime[i] for i in range(0, self.l1))
        flow_Z = sum((self.l0 + i + 1) * X_prime[i] for i in range(0, self.l1))
        s = self.compute_service_time(Z)
        self.T += s
        # update admission or eviction
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
        # update arrival
        Qe_prime = Qe_prime + self.lambda_rate * s

        # execute batch
        overflow_memory = sum((self.l0 + i) * X_prime[i] for i in range(0, self.l1))
        assert (overflow_memory <= self.B)
        completion = X_prime[self.l1-1]
        self.total_completion += completion
        X_prime_tmp = [0] * self.l1
        for i in range(1, self.l1):
            X_prime_tmp[i] = X_prime[i-1]

        self.X = X_prime_tmp
        self.Qe = Qe_prime
        self.Qr = Qr_prime
        self.n += 1
        print(f"n={self.n}, T={self.T:.3f}: after B, X={self.X}, Qe={self.Qe:.2f}, Qr={self.Qr}, completion={completion}")
        print(f"n={self.n}, T={self.T:.3f}: total_completion={self.total_completion}, throughput={self.total_completion/self.T}, lambda={self.lambda_rate}")
        
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
    l1 = 3
    B = 1400  # GPU容量
    X0 = [20, 20,20]  # 初始状态 (B/2, 0), 注意初始状态直接爆炸
    Qe0 = 1000  # 初始外部队列
    lambda_rate = 1999.1  # 到达率
    b0 = 0.1
    b1 = 0.01
    if_float = False
    
    # 创建模拟器
    sim = LLMSchedulerSimulator(l0, l1, B, X0, Qe0, lambda_rate, b0, b1, if_float)

    t = 10
    
    sim.run(t)
