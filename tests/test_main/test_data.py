"""Tests for the opt.data explainability interface."""

import math

import numpy as np
import pytest

from gradient_free_optimizers import (
    HillClimbingOptimizer,
    RandomSearchOptimizer,
)

from .test_optimizers._parametrize import optimizers_representative


def objective(para):
    return -(para["x"] ** 2 + para["y"] ** 2)


SEARCH_SPACE = {
    "x": np.linspace(-10, 10, 100),
    "y": np.linspace(-10, 10, 100),
}

N_ITER = 30


@pytest.fixture
def hill_opt():
    opt = HillClimbingOptimizer(SEARCH_SPACE)
    opt.search(objective, n_iter=N_ITER, verbosity=False)
    return opt


class TestDataAvailability:
    def test_available_after_search(self, hill_opt):
        assert hill_opt.data is not None

    def test_not_available_before_search(self):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        with pytest.raises(AttributeError, match="Call search"):
            _ = opt.data

    def test_search_data_backward_compat(self, hill_opt):
        sd = hill_opt.search_data
        assert len(sd) == N_ITER
        assert "score" in sd.columns

    def test_data_reflects_latest_search(self):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=10, verbosity=False)
        assert opt.data.n_iter == 10

        opt.search(objective, n_iter=20, verbosity=False)
        assert opt.data.n_iter == 20


class TestIterationCounts:
    def test_total_equals_init_plus_optimization(self, hill_opt):
        d = hill_opt.data
        assert d.n_iter == N_ITER
        assert d.n_init + d.n_optimization == d.n_iter

    def test_has_init_iterations(self, hill_opt):
        assert hill_opt.data.n_init > 0

    def test_single_iteration(self):
        opt = RandomSearchOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)
        d = opt.data
        assert d.n_iter == 1
        assert len(d.convergence_data) == 1
        assert len(d.results) == 1


class TestBestResults:
    def test_best_score(self, hill_opt):
        d = hill_opt.data
        assert d.best_score <= 0
        assert d.best_score == hill_opt.best_score

    def test_best_para(self, hill_opt):
        d = hill_opt.data
        assert isinstance(d.best_para, dict)
        assert "x" in d.best_para
        assert "y" in d.best_para

    def test_best_iteration_in_range(self, hill_opt):
        assert 0 <= hill_opt.data.best_iteration < N_ITER


class TestTiming:
    def test_times_non_negative(self, hill_opt):
        d = hill_opt.data
        assert d.total_time >= 0
        assert d.eval_time >= 0
        assert d.overhead_time >= 0

    def test_overhead_pct_in_range(self, hill_opt):
        assert 0 <= hill_opt.data.overhead_pct <= 100

    def test_eval_pct_in_range(self, hill_opt):
        assert 0 <= hill_opt.data.eval_pct <= 100

    def test_avg_times(self, hill_opt):
        d = hill_opt.data
        assert d.avg_eval_time >= 0
        assert d.avg_iter_time >= 0

    def test_throughput(self, hill_opt):
        assert hill_opt.data.throughput >= 0


class TestConvergence:
    def test_convergence_length(self, hill_opt):
        assert len(hill_opt.data.convergence_data) == N_ITER

    def test_convergence_monotonic(self, hill_opt):
        conv = hill_opt.data.convergence_data
        for i in range(1, len(conv)):
            assert conv[i] >= conv[i - 1]

    def test_score_improvements(self, hill_opt):
        assert hill_opt.data.n_score_improvements >= 1

    def test_last_improvement(self, hill_opt):
        d = hill_opt.data
        assert 0 <= d.last_improvement < N_ITER

    def test_longest_plateau(self, hill_opt):
        length, start, end = hill_opt.data.longest_plateau
        assert length >= 1
        assert start >= 0
        assert end < N_ITER
        assert end - start + 1 == length


class TestAcceptance:
    def test_rate_in_range(self, hill_opt):
        assert 0.0 <= hill_opt.data.acceptance_rate <= 1.0

    def test_n_proposed(self, hill_opt):
        assert hill_opt.data.n_proposed > 0

    def test_n_accepted(self, hill_opt):
        d = hill_opt.data
        assert d.n_accepted > 0
        assert d.n_accepted <= d.n_proposed


class TestScoreStatistics:
    def test_min_max_order(self, hill_opt):
        d = hill_opt.data
        assert d.score_min <= d.score_max

    def test_mean_in_range(self, hill_opt):
        d = hill_opt.data
        assert d.score_min <= d.score_mean <= d.score_max

    def test_std_non_negative(self, hill_opt):
        assert hill_opt.data.score_std >= 0

    def test_no_invalid(self, hill_opt):
        assert hill_opt.data.n_invalid == 0


class TestResults:
    def test_results_length(self, hill_opt):
        assert len(hill_opt.data.results) == N_ITER

    def test_results_structure(self, hill_opt):
        row = hill_opt.data.results[0]
        assert isinstance(row, dict)
        assert "score" in row
        assert "x" in row
        assert "y" in row


class TestRawData:
    def test_scores_proposed(self, hill_opt):
        assert len(hill_opt.data.raw.scores_proposed) == N_ITER

    def test_scores_accepted(self, hill_opt):
        raw = hill_opt.data.raw
        assert 0 < len(raw.scores_accepted) <= N_ITER

    def test_scores_best(self, hill_opt):
        assert len(hill_opt.data.raw.scores_best) > 0

    def test_positions_proposed(self, hill_opt):
        positions = hill_opt.data.raw.positions_proposed
        assert len(positions) == N_ITER
        assert isinstance(positions[0], dict)
        assert "x" in positions[0]

    def test_positions_accepted(self, hill_opt):
        positions = hill_opt.data.raw.positions_accepted
        assert len(positions) > 0
        assert isinstance(positions[0], dict)

    def test_eval_times(self, hill_opt):
        raw = hill_opt.data.raw
        assert len(raw.eval_times) == N_ITER
        assert all(t >= 0 for t in raw.eval_times)

    def test_iter_times(self, hill_opt):
        raw = hill_opt.data.raw
        assert len(raw.iter_times) == N_ITER
        assert all(t >= 0 for t in raw.iter_times)

    def test_convergence(self, hill_opt):
        assert len(hill_opt.data.raw.convergence) == N_ITER

    def test_improvement_iterations_sorted(self, hill_opt):
        iters = hill_opt.data.raw.improvement_iterations
        assert len(iters) > 0
        for i in range(1, len(iters)):
            assert iters[i] > iters[i - 1]

    def test_scores_all(self, hill_opt):
        assert len(hill_opt.data.raw.scores_all) == N_ITER


ALL_PRINT_SECTIONS = [
    "print_results",
    "print_search_stats",
    "print_statistics",
    "print_times",
]


class TestPrintSummary:
    def test_summary_box(self, capsys):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=20, verbosity=ALL_PRINT_SECTIONS)
        out = capsys.readouterr().out
        assert "Search Summary" in out
        assert "HillClimbingOptimizer" in out
        assert "objective" in out

    def test_summary_parameters(self, capsys):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=20, verbosity=ALL_PRINT_SECTIONS)
        out = capsys.readouterr().out
        assert "Best parameters:" in out
        assert "x:" in out
        assert "y:" in out

    def test_summary_sections(self, capsys):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=20, verbosity=ALL_PRINT_SECTIONS)
        out = capsys.readouterr().out
        assert "── Results ─" in out
        assert "── Search ─" in out
        assert "── Score Statistics ─" in out
        assert "── Timing ─" in out

    def test_summary_search_metrics(self, capsys):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=20, verbosity=ALL_PRINT_SECTIONS)
        out = capsys.readouterr().out
        assert "Accepted:" in out
        assert "Last improvement:" in out
        assert "Min:" in out
        assert "Max:" in out
        assert "Throughput:" in out

    def test_summary_subset_verbosity(self, capsys):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=20, verbosity=["print_results"])
        out = capsys.readouterr().out
        assert "Search Summary" in out
        assert "── Results ─" in out
        assert "── Timing ─" not in out


class TestAcrossOptimizers:
    @pytest.mark.parametrize(*optimizers_representative)
    def test_data_properties(self, Optimizer):
        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=20, verbosity=False)

        d = opt.data
        assert d.n_iter == 20
        assert d.n_init + d.n_optimization == 20
        assert len(d.convergence_data) == 20
        assert len(d.results) == 20
        assert d.best_score <= 0
        assert 0 <= d.best_iteration < 20
        assert d.n_score_improvements >= 1
        assert 0.0 <= d.acceptance_rate <= 1.0
        assert d.score_min <= d.score_max

    @pytest.mark.parametrize(*optimizers_representative)
    def test_raw_data(self, Optimizer):
        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=20, verbosity=False)

        raw = opt.data.raw
        assert len(raw.scores_proposed) == 20
        assert len(raw.scores_all) == 20
        assert len(raw.iter_times) == 20
        assert len(raw.convergence) == 20

    @pytest.mark.parametrize(*optimizers_representative)
    def test_summary_output(self, Optimizer, capsys):
        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=20, verbosity=ALL_PRINT_SECTIONS)
        out = capsys.readouterr().out
        assert "Search Summary" in out
        assert Optimizer.__name__ in out
