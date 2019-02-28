#!/usr/bin/env python

import yaml
import sys

def main(data_path):
    with open(data_path) as f:
        data = yaml.load(f)
    groups = data['groups']
    runners = data['runners']
    race = data['race']
    print_assignment([i for i in runners], runners, race)

def print_assignment(assignment, runners, race):
    '''
    assignment is an array of len(race) with a list of runners' names
    '''
    total_time = 0
    total_distance = 0

    name_w = max(max(len(i) for i in runners), 6)
    print('Num | Runner{:s} | Legs'.format(' ' * (name_w - 6)))
    print('----+-------{:s}-+--------------------------'.format('-' * (name_w - 6)))
    for i, name in enumerate(assignment):
        leg = race[i]
        pace = runners[name]['pace']
        distances = ' + '.join('{:4.1f} mi'.format(i) for i in leg)
        times = ' + '.join('{:3.0f} min'.format(i*pace / 60) for i in leg)
        distance = sum(leg)
        time = sum(leg) * pace / 60
        pace_str = '{:d}:{:02d}'.format(pace // 60, pace % 60).ljust(5)
        print('{:3d} | {:s} | {:s} = {:4.1f} mi'.format(i+1, name.ljust(name_w), distances, distance))
        print('    | {:s}{:s} | {:s} = {:3.0f} min'.format(pace_str, ' '*(name_w-5), times, time))
        if i != len(assignment)-1:
            print('    | {:s} |'.format(' '*name_w))

        total_time += time
        total_distance += distance

    print('')
    print('Total distance: {:.1f} mi'.format(total_distance))
    print('Total time: {:.0f} hr {:.0f} min'.format(total_time / 60, total_time % 60))

if __name__ == '__main__':
    main(sys.argv[1])
