import argparse

from datetime import datetime
from sgp4.api import jday
import torch


def main(args: argparse.Namespace):
    from screener.propagator.sgp4 import JAXSGP4Propagator

    propagator = JAXSGP4Propagator()

    tles = []

    # Load TLEs from the input file
    with open(args.input_file, 'r') as f:
        for _ in range(args.nums if args.nums > 0 else float('inf')):
            _ = f.readline()  # Skip the satellite name line
            line1 = f.readline()
            line2 = f.readline()
            if not line1 or not line2:
                break
            tles.append((line1.strip(), line2.strip()))

    unified_epoch_utc = args.unified_epoch_utc
    t = datetime.strptime(unified_epoch_utc, "%Y-%m-%d %H:%M:%S UTC")

    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)

    rs, vs = propagator.propagate(
        tle_batch=tles,
        julian_date_start=(jd, fr),
        julian_date_end=(jd, fr),
        time_mesh=60,
    )
    torch.save((rs, vs), args.output_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Propagate TLEs to a unified epoch and save the results.")
    parser.add_argument(
        "--input_file",
        '-i',
        type=str,
        required=True,
        help="Path to the input file containing TLEs.",
    )
    parser.add_argument(
        "--output_file",
        '-o',
        type=str,
        required=True,
        help="Path to the output file to save propagated results.",
    )
    parser.add_argument(
        "--unified_epoch_utc",
        '-t',
        type=str,
        required=True,
        help=
        "unified utc time for propagation in format 'YYYY-MM-DD HH:MM:SS UTC'.",
    )
    parser.add_argument(
        "--nums",
        type=int,
        default=-1,
        help="Number of TLEs to process. Default is -1 (process all).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
