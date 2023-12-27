import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--gpu",type=int,default='1')
parser.add_argument("--history_dim",type=int, default=5)
parser.add_argument("--client_nums",type=int)

args = parser.parse_args()

print(args.history_dim)
print(args.client_numss)