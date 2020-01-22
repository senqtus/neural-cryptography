from argparse import ArgumentParser
from src.model import CryptoNet
from src.config import *
from src.communicate_net import CommunicateNet
from src.eve_net import EveNet
import torch 
import numpy as np

def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--msg-len', type=int,
                        dest='msg_len', help='message length',
                        metavar='MSG_LEN', default=MSG_LEN)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='Number of Epochs in Adversarial Training',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--iterations', type=int,
                        dest='iterations', help='Number of iterations in epoch',
                        metavar='ITERATIONS', default=NUM_ITERATIONS)

    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    crypto_net = CryptoNet(msg_len=options.msg_len, epochs=options.epochs,
                           batch_size=options.batch_size, learning_rate=options.learning_rate,
                           iterations=options.iterations)
    crypto_net.train()
    crypto_net.plot_errors()

#TODO
def run():
    alice = CommunicateNet()
    alice.load_state_dict(torch.load("neural-cryptography/weights/alice"))
    alice.eval()
    bob = CommunicateNet()
    bob.load_state_dict(torch.load("neural-cryptography/weights/bob"))
    bob.eval()
    eve = EveNet()
    eve.load_state_dict(torch.load("neural-cryptography/weights/eve"))
    eve.eval()
    key=np.random.randint(0, 2, size=(1,16)) * 2 - 1
    msg=np.random.randint(0, 1, size=(1,16))
    key = torch.tensor(key, dtype=torch.float)
    msg = torch.tensor(msg, dtype=torch.float)
    alice_input = torch.cat((msg, key), 1)
    alice_output = alice(alice_input).view(1,16).detach()
    bob_input = torch.cat((alice_output, key), 1)
    bob_output=bob(bob_input)
    print(alice_input)
    print(bob_output)

if __name__ == '__main__':
    main()
