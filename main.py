import argparse
import os
import sys
import Config as cfg
from NeuralNet import NeuralNet
from Datasets import Datasets


parser = argparse.ArgumentParser(description='Gil Shomron, gilsho@campus.technion.ac.il',
                                 formatter_class=argparse.RawTextHelpFormatter)

model_names = list(cfg.MODELS.keys())

parser.add_argument('-a', '--arch', metavar='ARCH', choices=model_names, required=True,
                    help='model architectures and datasets:\n' + ' | '.join(model_names))
parser.add_argument('--action', choices=['QUANTIZE', 'INFERENCE'], required=True,
                    help='QUANTIZE: symmetric min-max uniform quantization\n'
                         'INFERENCE: either regular inference or hardware simulated inference')
parser.add_argument('--desc')
parser.add_argument('--chkp', default=None, metavar='PATH',
                    help='model checkpoint')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--x_bits', default=None, type=int, metavar='N',
                    help='activations quantization bits')
parser.add_argument('--w_bits', default=None, type=int, metavar='N',
                    help='weights quantization bits')
parser.add_argument('--threads', choices=['4', '2', '1'], default=1,
                    help='number of default threads')
parser.add_argument('--debug', action='store_true',
                    help='enable debugging of backprop')
parser.add_argument('--gpu', nargs='+', default=None,
                    help='GPU to run on (default: 0)')
parser.add_argument('-v', '--verbosity', default=0, type=int,
                    help='verbosity level (0,1,2) (default:0)')
parser.add_argument('--layer_to_reduce', default=None, type=int, metavar='N')
parser.add_argument('--bits_to_allow', default=None, type=int, metavar='N')
parser.add_argument('--bits_to_all', default=None, type=int, metavar='N')

def quantize_network(arch, dataset, train_gen, test_gen, model_chkp=None, only_stats=False,
                     x_bits=8, w_bits=8, desc=None, n=0):

    nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

    nn.model.set_quantize(True)
    nn.model.set_quantization_bits(x_bits, w_bits)
    nn.model.set_unfold(False)
    nn.model.set_custom_matmul(False)
    nn.model.set_min_max_update(True)
    #added but no use
    #nn.model.set_bitsCounter(n)
    #added but no use
    nn.best_top1_acc = 0
    nn.next_train_epoch = 0

    nn.train(train_gen, test_gen, epochs=1, lr=0, iterations=2048 / cfg.BATCH_SIZE)

    return


def inference(arch, dataset, train_gen, test_gen, model_chkp, x_bits=None, w_bits=None, threads=1,
              unfold=False, custom_matmul=False, desc=None, layer_num=None, bits_allowed=32,bits_to_all=20):

    nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

    nn.model.set_quantize(x_bits is not None and w_bits is not None)
    nn.model.set_quantization_bits(x_bits, w_bits)
    nn.model.set_unfold(unfold)
    nn.model.set_custom_matmul(custom_matmul)
    nn.model.set_min_max_update(False)
    #added
    nn.model.set_bitsAllowed(layer_num,bits_allowed,bits_to_all)
    nn.model.set_layer_number()
    nn.model.set_test_counter()
    #nn.model.set_array()
    #added


    # FOR BATCH-NORM
    # nn.best_top1_acc = 0
    # nn.next_train_epoch = 0
    # nn.train(train_gen, test_gen, epochs=1, lr=0, iterations=2048 / cfg.BATCH_SIZE)
    # FOR BATCH-NORM

    nn.test(test_gen)
    return


def main():
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    cfg.BATCH_SIZE = args.batch_size
    cfg.VERBOSITY = args.verbosity
    cfg.USER_CMD = ' '.join(sys.argv)
    cfg.DEBUG = args.debug

    arch = args.arch.split('_')[0]
    dataset = args.arch.split('_')[1]
    dataset_ = Datasets.get(dataset)

    # Deterministic random numbers
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # np.random.seed(1)

    test_gen, _ = dataset_.testset(batch_size=args.batch_size)
    (train_gen, _), (_, _) = dataset_.trainset(batch_size=args.batch_size, max_samples=None, random_seed=16)

    model_chkp = None if args.chkp is None else cfg.RESULTS_DIR + '/' + args.chkp

    #our new flags
    #limit specefic layer
    layer_num = args.layer_to_reduce
    #limit the allowed bits
    bits_allowed = args.bits_to_allow
    #limit the total bits
    bits_to_all = args.bits_to_all

    #FOR DEFAULT BITS:
    if args.bits_to_allow == None:
        args.bits_to_allow = 32
    if args.bits_to_all == None:
        args.bits_to_all = 32

    if args.action == 'QUANTIZE':
        quantize_network(arch, dataset, train_gen, test_gen,
                         model_chkp=model_chkp,
                         only_stats=True, x_bits=args.x_bits, w_bits=args.w_bits, desc=args.desc)

    elif args.action == 'INFERENCE':
        #for loop for running a bunch of tests
        for i in range(17,18): #from 18 to 18
            args.bits_to_all = i
            for k in range(16, i+1):  # from 0 to i
                args.bits_to_allow = k

        # for k in range (16,19): #from 16 to 18 for every layer
        #     args.bits_to_all = k
        #     for i in range (14,20): #from 14 to 19 for specific layer
        #         args.bits_to_allow = i

                inference(arch, dataset, train_gen, test_gen,
                          model_chkp=model_chkp,
                          x_bits=args.x_bits, w_bits=args.w_bits,
                          threads=int(args.threads), unfold=True, custom_matmul=True, #TRUE & TRUE (FOR CUSTOM CUDA)
                          desc=args.desc,layer_num=args.layer_to_reduce,bits_allowed=args.bits_to_allow,bits_to_all=args.bits_to_all)

    return


if __name__ == '__main__':
    main()
