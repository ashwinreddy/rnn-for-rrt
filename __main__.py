import argparse
from train import trainNetwork
from test import test

parser = argparse.ArgumentParser()
parser.add_argument("-train", action='store_true')
parser.add_argument("-test", action='store_true')
args = parser.parse_args()


def main():
	if args.train:
		trainNetwork()

	if args.test:
		test()


if __name__ == "__main__":
	main()