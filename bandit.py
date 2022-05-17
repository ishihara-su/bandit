# bandit.py

import math
import matplotlib.pyplot as plt
import random


class Bandit:
    def __init__(self, seed: int = 1, prob: float = 0.3) -> None:
        self.prob = prob
        self.rg = random.Random(seed)

    def spin(self) -> bool:
        r = self.rg.random()
        if r <= self.prob:
            return True
        return False


class BanditManager:
    def __init__(self, seed=1, n_bandits=2, high_prob=0.3, low_prob=0.05, mean_interval: float = 1500.0) -> None:
        seed_offset = 713
        self.rg = random.Random(seed)
        self.bandits = [Bandit(seed+seed_offset+i) for i in range(n_bandits)]
        self.high_prob = high_prob
        self.low_prob = low_prob
        self.count = 0
        self.mean_interval = mean_interval
        self._change_good_one()
        self.next_update = self.count + \
            math.ceil(self.rg.expovariate(1.0/self.mean_interval))

    def update(self):
        self.count += 1
        if self.count >= self.next_update:
            self._change_good_one()
            self.next_update = self.count + \
                math.ceil(self.rg.expovariate(1.0/self.mean_interval))

    def _change_good_one(self):
        good_one = self.rg.randrange(len(self.bandits))
        for i in range(len(self.bandits)):
            prob = self.high_prob if i == good_one else self.low_prob
            self.bandits[i].prob = prob


class Agent:
    """Agent - Default Agent
    Always selects an agent randomly
    """

    def __init__(self, bandits: list[Bandit]) -> None:
        self.score = 0
        self.score_hist = []
        self.selection_hist = []
        self.count = 0
        self.bandits = bandits

    def act(self) -> None:
        b = self.select()
        result = self.bandits[b].spin()
        if result:
            self.score += 1
        self.count += 1
        self.score_hist.append(self.score)
        self.selection_hist.append(b)
        self.post_process(b, result)

    def select(self) -> int:
        return random.randrange(len(self.bandits))

    def post_process(self, b: int, result: bool):
        pass


class ObstinateAgent(Agent):
    def __init__(self, bandits: list[Bandit]) -> None:
        super().__init__(bandits)
        self.selected_bandit = random.randrange(len(self.bandits))

    def select(self) -> int:
        return self.selected_bandit


class HitCounter:
    def __init__(self, size: int = 50) -> None:
        self.array = [0] * size
        self.pos = 0
        self.size = size

    def put(self, hit_result: bool) -> None:
        self.array[self.pos] = 1 if hit_result else 0
        self.pos = (self.pos + 1) % self.size

    def hit_ratio(self) -> float:
        return self.array.count(1) / self.size


class SmartAgent1(Agent):
    def __init__(self, bandits: list[Bandit], initial_probs: list[float],
                 forgetting_rate: float = 0.1, memory_length: int = 50,
                 random_ratio: float = 0.1) -> None:
        self.probs = []
        self.hist_probs = []
        self.selection_count = [0] * len(bandits)
        for p in initial_probs:
            if p < 0.0 or 1.0 < p:
                raise ValueError('Prob is out of range. (0 <= prob <= 1.0)')
            self.probs.append(p)
            self.hist_probs.append([])
        if forgetting_rate < 0.0 or 1.0 < forgetting_rate:
            raise ValueError('Forgetting rate is of range (0 <= rate <= 1.0')
        self.forgetting_rate = forgetting_rate
        if memory_length < 1:
            raise ValueError('Memory length must be positive.')
        self.hit_counters = [HitCounter(memory_length)
                             for i in range(len(bandits))]
        if random_ratio < 0.0 or 1.0 < random_ratio:
            raise ValueError('Random ratio is of range (0 <= rate <= 1.0')
        self.random_ratio = random_ratio
        super().__init__(bandits)

    def select(self) -> int:
        if random.random() <= self.random_ratio:
            return random.randrange(len(self.bandits))
        maxp = -1.0
        selected = -1
        for b in range(len(self.bandits)):
            if self.probs[b] > maxp:
                selected = b
                maxp = self.probs[b]
        return selected

    def post_process(self, b: int, result: bool) -> None:
        self.selection_count[b] += 1
        self.hit_counters[b].put(result)
        for i in range(len(self.bandits)):
            self.probs[i] = ((1.0 - self.forgetting_rate) * self.probs[i] +
                             self.forgetting_rate * self.hit_counters[i].hit_ratio())
            self.hist_probs[i].append(self.probs[i])


class Simulator:
    def __init__(self, bandit_manager: BanditManager, agent: Agent) -> None:
        self.bandit_manager = bandit_manager
        self.agent = agent

    def run(self, n_steps: int = 10000):
        for i in range(n_steps):
            self.agent.act()
            self.bandit_manager.update()


def main() -> None:
    n_steps = 10000
    seed = 13
    n_bandits = 4
    mean_interval = 500

    bm1 = BanditManager(seed, n_bandits, mean_interval=mean_interval)
    random_agent = Agent(bm1.bandits)
    sim1 = Simulator(bm1, random_agent)
    sim1.run(n_steps)

    bm2 = BanditManager(seed, n_bandits, mean_interval=mean_interval)
    obstinate_agent = ObstinateAgent(bm2.bandits)
    sim2 = Simulator(bm2, obstinate_agent)
    sim2.run(n_steps)

    bm3 = BanditManager(seed, n_bandits, mean_interval=mean_interval)
    smart_agent1 = SmartAgent1(
        bm3.bandits, [1.0/n_bandits] * n_bandits, 0.2, 50, 0.3)
    sim3 = Simulator(bm3, smart_agent1)
    sim3.run(n_steps)

    fig, (a_score, a_prob) = plt.subplots(nrows=2)
    x = [i for i in range(n_steps)]
    a_score.step(x, random_agent.score_hist, linewidth=1, label='Random')
    a_score.step(x, obstinate_agent.score_hist, linewidth=1, label='Obstinate')
    a_score.step(x, smart_agent1.score_hist, linewidth=1, label='Smart1')
    a_score.legend()
    a_score.set_ylabel('Score')

    for i in range(n_bandits):
        a_prob.step(x, smart_agent1.hist_probs[i], linewidth=1, label=str(i))
    a_prob.legend()
    a_prob.set_ylabel('Prob.')
    plt.show()


if __name__ == '__main__':
    main()
