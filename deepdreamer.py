###############################################################################
# Deep Dreamer
# Author: Kesara Rathnayake ( kesara [at] kesara [dot] lk )
###############################################################################

from argparse import ArgumentParser
import sys

from deepdreamer.deepdreamer import deepdream, list_layers


def main():
    try:
        parser = ArgumentParser(description="Deep dreamer")
        parser.add_argument(
            "--zoom", choices=["true", "false"], default="true",
            help="zoom dreams (default: true)")
        parser.add_argument(
            "--scale", type=float, default=0.05,
            help="scale coefficient for zoom (default: 0.05)")
        parser.add_argument(
            "--dreams", type=int, default=100,
            help="number of images (default: 100)")
        parser.add_argument(
            "--itern", type=int, default=10,
            help="dream iterations (default: 10")
        parser.add_argument(
            "--octaves", type=int, default=4,
            help="dream octaves (default: 4)")
        parser.add_argument(
            "--octave-scale", type=float, default=1.4,
            help="dream octave scale (default: 1.4)")
        parser.add_argument(
            "--layers", type=str, default="inception_4c/output",
            help="dream layers (default: inception_4c/output)")
        parser.add_argument(
            "--clip", choices=["true", "false"], default="true",
            help="clip dreams (default: true)")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("image", nargs="?")
        group.add_argument("--list-layers", action="store_true", help="list layers")
        args = parser.parse_args()
        if args.list_layers:
            list_layers()
        else:
            zoom = True
            if args.zoom == "false":
                zoom = False
            clip = True
            if args.clip == "false":
                clip = False
            deepdream(
                args.image, zoom=zoom, scale_coefficient=args.scale,
                irange=args.dreams, iter_n=args.itern, octave_n=args.octaves,
                octave_scale=args.octave_scale, end=args.layers, clip=clip)
    except Exception as e:
        print("Error: {}".format(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
