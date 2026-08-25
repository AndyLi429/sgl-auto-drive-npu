#!/usr/bin/env python3
"""Compute a reproducible single-memory-roof roofline summary."""

import argparse
import json


def positive(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute arithmetic intensity and an optional roofline bound."
    )
    parser.add_argument("--flops", required=True, type=positive, help="Executed FLOPs")
    parser.add_argument("--bytes", required=True, type=positive, help="Bytes moved")
    parser.add_argument("--peak-tflops", type=positive, help="Compute peak in decimal TFLOPS")
    parser.add_argument("--peak-gbs", type=positive, help="Bandwidth peak in decimal GB/s")
    args = parser.parse_args()

    if (args.peak_tflops is None) != (args.peak_gbs is None):
        parser.error("--peak-tflops and --peak-gbs must be provided together")

    intensity = args.flops / args.bytes
    result = {
        "flops": args.flops,
        "bytes": args.bytes,
        "arithmetic_intensity_flop_per_byte": intensity,
    }

    if args.peak_tflops is not None:
        peak_flops = args.peak_tflops * 1e12
        peak_bytes_s = args.peak_gbs * 1e9
        memory_roof_flops = intensity * peak_bytes_s
        roofline_flops = min(peak_flops, memory_roof_flops)
        result.update(
            {
                "peak_tflops": args.peak_tflops,
                "peak_gbs": args.peak_gbs,
                "machine_balance_flop_per_byte": peak_flops / peak_bytes_s,
                "memory_roof_tflops": memory_roof_flops / 1e12,
                "roofline_tflops": roofline_flops / 1e12,
                "bound": "memory" if memory_roof_flops < peak_flops else "compute",
                "compute_time_seconds": args.flops / peak_flops,
                "memory_time_seconds": args.bytes / peak_bytes_s,
                "lower_bound_seconds": max(
                    args.flops / peak_flops, args.bytes / peak_bytes_s
                ),
            }
        )

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
