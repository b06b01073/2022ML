import argparse
import math

file_path = './testfile.txt'

def parse_outcome(outcome):
    head_count = 0
    tail_count = 0

    for i in outcome:
        if i == '1':
            head_count += 1
        elif i == '0':
            tail_count += 1
    return head_count, tail_count


# return the gamma in nature log
def ln_gamma(k):
    if k == 1 or k == 2:
        return 0
    res = 0
    for i in range(1, k):
        res += math.log(i)
    return res



# use log to prevent overflow
def beta(a, b, p):
    return math.exp((a - 1) * math.log(p) + (b - 1) * math.log(1 - p) + ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b))


# return the factorial in nature log
def ln_factorial(a):
    if a == 0:
        return 1
    res = 0
    for i in range(1, a + 1):
        res += math.log(i)

    return res


def bin_likelihood(trial, success, p):
    failure = trial - success

    return math.exp(ln_factorial(trial) - ln_factorial(failure)- ln_factorial(success) + success * math.log(p) + failure * math.log(1- p))

def main(is_float=False):
    a = input('a = ')
    b = input('b = ')

    a = float(a) if is_float else int(a)
    b = float(b) if is_float else int(b)

    

    with open(file_path) as f:
        for case, outcome in enumerate(f.readlines(), start=1):
            head_count, tail_count = parse_outcome(outcome)
            p = head_count / (head_count + tail_count)
            likelihood = bin_likelihood(head_count + tail_count, head_count, p)

            print(f'case {case}: {outcome[:-1]}')
            print(f'Likelihood: {likelihood}')
            print(f'Beta prior: a = {a} b = {b}')

            a += head_count
            b += tail_count
            print(f'Beta posterior: a = {a} b = {b}\n')


            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--float", "-f", help="set the datatype of a and b to float", type=bool, default=False, choices=[True, False])
    args = parser.parse_args()
    main(args.float)