"""
Các bản vá nhỏ quanh neat-python 2.0.0 cho pipeline NEAT của nhóm
===================================================================
Ba thứ neat-python 2.0.0 không tự làm, đã kiểm thử trong báo cáo (mục 8.13):

1. unify_initial_hidden_keys(population, config)
   Khi num_hidden > 0, neat-python cấp cho nút ẩn của MỖI genome ban đầu một khóa riêng
   (genome 1 có nút 2, genome 2 có nút 3, ...). Khoảng cách phân loài so gen theo khóa,
   nên mọi genome trông như khác hẳn nhau, mỗi genome thành một loài và quần thể sụp
   (lỗi "population size is less than num_species * min_species_size").
   Gọi một lần ngay sau neat.Population(config).

2. TargetSpeciesReporter(config, target=8)
   Ngưỡng tương thích cố định 3.0 chỉ tạo 1-2 loài với 15 đầu vào, vì neat-python chia
   khoảng cách cho số gen (30 kết nối ban đầu) nên khác biệt cấu trúc bị pha loãng.
   Reporter này tăng/giảm ngưỡng 0.1 sau mỗi thế hệ để giữ khoảng `target` loài
   (cách làm của rtNEAT, SharpNEAT, goNEAT; neat-python không có tùy chọn này).

3. ValidationChampion
   population.run() trả về genome có fitness huấn luyện cao nhất, mà fitness này ồn vì
   mỗi thế hệ chỉ chạy vài bản đồ ngẫu nhiên. Lớp này chấm top-k genome trên bản đồ
   validation định kỳ và giữ genome có tỉ lệ tới đích trung bình cao nhất.

Cách dùng trong NEATRunner.run():

    population = neat.Population(self.neat_config)
    unify_initial_hidden_keys(population, self.neat_config)
    population.add_reporter(TargetSpeciesReporter(self.neat_config, target=8))
    ...
    # trong _eval_genomes, sau khi gán fitness:
    self.champion.update(self._gen, [g for _, g in genomes])
    ...
    winner = self.champion.best_genome   # thay cho giá trị population.run() trả về
"""

from __future__ import annotations

import copy
from itertools import count
from typing import Callable, Iterable, Sequence

import neat
import neat.reporting


def unify_initial_hidden_keys(population: neat.Population, config: neat.Config) -> None:
    """Đánh lại khóa nút ẩn của quần thể ban đầu thành num_outputs, num_outputs+1, ... rồi phân loài lại."""
    gc = config.genome_config
    if gc.num_hidden <= 0:
        return
    n_out = gc.num_outputs
    for g in population.population.values():
        hidden = sorted(k for k in g.nodes if k >= n_out)
        mapping = {old: n_out + i for i, old in enumerate(hidden)}
        nodes, conns = {}, {}
        for k, ng in g.nodes.items():
            ng.key = mapping.get(k, k)
            nodes[ng.key] = ng
        for (a, b), cg in g.connections.items():
            cg.key = (mapping.get(a, a), mapping.get(b, b))
            conns[cg.key] = cg
        g.nodes, g.connections = nodes, conns
    # nút mới về sau bắt đầu từ khóa chưa dùng
    gc.node_indexer = count(n_out + gc.num_hidden)
    population.species = config.species_set_type(config.species_set_config, population.reporters)
    population.species.speciate(config, population.population, population.generation)


class TargetSpeciesReporter(neat.reporting.BaseReporter):
    """Điều chỉnh compatibility_threshold để số loài bám quanh `target`."""

    def __init__(self, config: neat.Config, target: int = 8, step: float = 0.1, min_threshold: float = 0.3):
        self.sc = config.species_set_config
        self.target, self.step, self.min_threshold = target, step, min_threshold

    def post_evaluate(self, config, population, species, best_genome):
        n = len(species.species)
        thr = self.sc.compatibility_threshold
        if n < self.target:
            thr -= self.step
        elif n > self.target:
            thr += self.step
        self.sc.compatibility_threshold = max(self.min_threshold, thr)


class ValidationChampion:
    """Giữ genome tốt nhất theo tỉ lệ tới đích trên bản đồ validation.

    score_fn(genome) -> float trả về SR trung bình trên các cấp validation
    (ví dụ: trung bình SR của 25 bản đồ val mỗi cấp, mạng reset trạng thái mỗi tập).
    """

    def __init__(self, score_fn: Callable[[neat.DefaultGenome], float], every: int = 10, top_k: int = 5):
        self.score_fn, self.every, self.top_k = score_fn, every, top_k
        self.best_genome = None
        self.best_score = -1.0
        self.best_generation = None

    def update(self, generation: int, genomes: Iterable[neat.DefaultGenome]) -> float | None:
        if generation % self.every != 0:
            return None
        top: Sequence = sorted(genomes, key=lambda g: g.fitness, reverse=True)[: self.top_k]
        best_here = -1.0
        for g in top:
            s = float(self.score_fn(g))
            best_here = max(best_here, s)
            if s >= self.best_score:
                self.best_score, self.best_genome, self.best_generation = s, copy.deepcopy(g), generation
        return best_here
