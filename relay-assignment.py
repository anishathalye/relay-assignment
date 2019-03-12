#!/usr/bin/env python

import yaml
import numpy as np
import scipy.optimize
import sys


class DisjointSet:
    def __init__(self):
        self._leader = {}
        self._group = {}

    def add(self, elem):
        if self._leader.get(elem) is None:
            self._leader[elem] = elem
            self._group[elem] = {elem}

    def merge(self, a, b):
        la = self._leader[a]
        lb = self._leader[b]
        if la is None or lb is None:
            raise ValueError('must be added to set first')
        ga = self._group[la]
        gb = self._group[lb]
        if len(ga) < len(gb):
            a, la, ga, b, lb, gb = b, lb, gb, a, la, ga
        ga |= gb
        del self._group[lb]
        for elem in gb:
            self._leader[elem] = la

    def group(self, elem):
        return self._group[self._leader[elem]]

    def groups(self):
        return [list(self._group[i]) for i in self._leader if self._leader[i] == i]


def optimal_assignment(groups, runners, race):
    '''
    Constructs an optimal assignment as follows:
    1. groups are not broken up
    2. happiness is maximized (defined as sum of rank of assignments)
    3. time is minimized

    This optimization can be done by enumerating over possible group
    assignments to groups of the race and then using the Hungarian algorithm in
    the inner loop.
    '''

    # merge overlaps in groups

    ds = DisjointSet()
    for r in runners:
        ds.add(r)
    for group in groups:
        for elem in group[1:]:
            ds.merge(group[0], elem)
    groups = [i for i in ds.groups() if len(i) > 1]

    # list all possible group assignments
    #
    # all_group_assignments[_][i] = j ==> group i is in race group j

    all_group_assignments = [()]
    if groups is None:
        groups = []
    for i in range(len(groups)):
        all_group_assignments = [(i,) + e for e in all_group_assignments for i in range(len(race['groups']))]

    # make cost matrix
    #
    # symbolically, our costs for runner r being placed to run leg i are
    # (rank(r, i), time(r, i)), where rank(r, i) is the rank given to leg i,
    # and time(r, i) is the time it would take to run leg i. sums are done as
    # (x, y) + (z, w) = (x + z, y + w) and comparisons are done the same as
    # Python tuple comparisons.

    index_to_runner = {i: name for i, name in enumerate(runners)} # fix mapping
    runner_to_index = {name: i for i, name in index_to_runner.items()}
    N = len(runners)
    def rank(r, i):
        return runners[index_to_runner[r]]['ranking'].index(i+1)
    def time(r, i):
        return int(runners[index_to_runner[r]]['pace'] * sum(race['legs'][i]))
    symbolic_C = [
        [
            (rank(r, i), time(r, i)) for i in range(N)
        ] for r in range(N)
    ]

    # to encode this into real numbers, we encode (x, y) -> big*x + y, where
    # big is some big number chosen appropriately.
    big = int(10**np.ceil(np.log10(2 * N * max(max(i[1] for i in r) for r in symbolic_C))))

    C = np.zeros((N, N), dtype=np.int64)
    for r in range(N):
        for i in range(N):
            sym = symbolic_C[r][i]
            C[r][i] = sym[0]*big + sym[1]

    # try all possibilities, creating a cost matrix representing this
    # possibility by marking illegal assignments
    inf = 2 * N * N * big
    best = None
    best_cost = inf
    for possibility in all_group_assignments:
        C_masked = C.copy()
        # mask out relevant entries
        for ig, g in enumerate(groups):
            ok = race['groups'][possibility[ig]]
            for mem in g:
                for i in range(N):
                    if i+1 not in ok:
                        C_masked[runner_to_index[mem]][i] = inf
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(C_masked)
        cost = C_masked[row_ind, col_ind].sum()
        if cost < inf and cost < best_cost:
            best_cost = cost
            best = [{i: index_to_runner[r] for r, i in zip(row_ind, col_ind)}[i] for i in range(N)]

    return best


def greedy_by_pace_assignment(groups, runners, race):
    available = set(range(len(race['legs'])))
    assignment = {}
    for name, r in sorted(runners.items(), key=lambda i: i[1]['pace']):
        for pref in r['ranking']:
            if pref-1 in available:
                assignment[pref-1] = name
                available.remove(pref-1)
                break
    return [assignment[i] for i in range(len(race['legs']))]


def greedy_by_time_assignment(groups, runners, race):
    available = set(range(len(race['legs'])))
    assignment = {}
    runners_by_pace = sorted(runners.items(), key=lambda i: i[1]['pace'])
    matching = zip(runners_by_pace, sorted(enumerate(race['legs']), key=lambda i: sum(i[1]), reverse=True))
    assignment = {i[1][0]: i[0][0] for i in matching}
    return [assignment[i] for i in range(len(race['legs']))]


def print_assignment(assignment, runners, race, only_summary=False):
    '''
    Nicely formats and prints an assignment of runners to legs of a race.

    assignment: an array of len(race) with a list of runners' names.
    '''
    total_time = 0
    total_distance = 0
    total_rank = 0

    name_w = max(max(len(i) for i in runners), 6)
    if not only_summary:
        print('Num | Runner{:s} | Legs'.format(' ' * (name_w - 6)))
        print('----+-------{:s}-+--------------------------'.format('-' * (name_w - 6)))
    for i, name in enumerate(assignment):
        leg = race['legs'][i]
        rank = runners[name]['ranking'].index(i+1)+1
        pace = runners[name]['pace']
        distances = ' + '.join('{:4.1f} mi'.format(i) for i in leg)
        times = ' + '.join('{:3.0f} min'.format(i*pace / 60) for i in leg)
        distance = sum(leg)
        time = sum(leg) * pace / 60
        pace_str = '{:d}:{:02d}'.format(pace // 60, pace % 60).ljust(5)
        if not only_summary:
            print('{:3d} | {:s} | {:s} = {:4.1f} mi'.format(i+1, name.ljust(name_w), distances, distance))
            print('#{:2d} | {:s}{:s} | {:s} = {:3.0f} min'.format(rank, pace_str, ' '*(name_w-5), times, time))
            if i != len(assignment)-1:
                print('    | {:s} |'.format(' '*name_w))

        total_time += time
        total_distance += distance
        total_rank += rank

    if not only_summary:
        print('')
        print('Total distance: {:.1f} mi'.format(total_distance))
    print('Total time: {:d} hr {:d} min'.format(int(total_time)//60, int(total_time) % 60))
    avg_pace = total_time / total_distance
    print('Average pace: {:d}:{:02d} min/mi'.format(int(avg_pace), int(avg_pace * 60) % 60))
    print('Sum of ranks: {:d}'.format(total_rank))


def main(data_path):
    with open(data_path) as f:
        data = yaml.load(f)
    groups = data['groups']
    runners = data['runners']
    race = data['race']
    print('=== Minimum Time ===\n')
    print_assignment(greedy_by_time_assignment(groups, runners, race), runners, race, only_summary=True)
    print('\n=== Greedy by Pace ===\n')
    print_assignment(greedy_by_pace_assignment(groups, runners, race), runners, race, only_summary=True)
    print('\n=== Optimal ===\n')
    print_assignment(optimal_assignment(groups, runners, race), runners, race)


if __name__ == '__main__':
    main(sys.argv[1])
