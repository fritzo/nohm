import argparse
from timeit import default_timer

from .runtime import parse, readback, reduce


def factorial(args):
    """
    See Asperti98 ss 9.5 pp 296.
    This uses a slightly simplified form of nextfact.
    """
    rows = []
    print("n\tparse\treduce\treadback")
    print("-" * 8 * 4)
    for n in range(1 + args.max_size):
        text = f"""
        LET id LAM x x
        LET succ LAM n LAM f LAM x APP APP n f APP f x
        LET mult LAM m LAM n LAM f APP m APP n x
        LET pair LAM x LAM y LAM f APP APP f x y
        LET fst LAM x LAM y x
        LET snd LAM x LAM y y
        LET nextfact LAM xy
          APP xy LAM x LAM y
          LET y1 APP succ y
          APP pair APP APP mult x y1 y1
        LET factorial LAM n APP APP APP n nextfact APP APP pair 1 0 fst
        APP APP APP factorial {n} id id
        """
        t0 = default_timer()
        main = parse(text)
        t1 = default_timer()
        reduce(main)
        t2 = default_timer()
        result = readback(main)
        t3 = default_timer()
        assert result == "LAM a a"
        row = (n, t1 - t0, t2 - t1, t3 - t2)
        rows.append(row)
        print("{}\t{:0.4g}\t{:0.4g}\t{:0.4g}".format(*row))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark reduction")
    parser.add_argument("-b", "--benchmark", default="factorial")
    parser.add_argument("-n", "--max-size", default=10, type=int)
    args = parser.parse_args()
    benchmark = globals()[args.benchmark]
    benchmark(args)


if __name__ == "__main__":
    main()
