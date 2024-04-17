import argparse

def create_parser(cfg):
    parser = argparse.ArgumentParser(description='PyTorch PULP-DroNet Training')
    parser.add_argument('-d', '--data_path', help='path to dataset',
                        default=cfg.data_path)
    parser.add_argument('-m', '--model_name', default=cfg.model_name,
                        help='model name that is created when training')
    parser.add_argument('--epochs', default=cfg.epochs, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=cfg.batch_size, type=int,
                        help='mini-batch size (default: 32), this is the total '
                            'batch size of all GPUs')
    parser.add_argument('--gpu', help='which gpu to use. Just one at'
                        'the time is supported', default=cfg.gpu)
    parser.add_argument('-j', '--workers', default=cfg.workers, type=int, metavar='N',
                        help='number of data loading workers (default: 4)') 
    parser.add_argument('--verbose', action='store_true', help='verbose prints on')
    return parser


class cfg:
    pass

# default for all scripts
cfg.data_path='./dataset/'
cfg.model_name='model'
cfg.epochs=100
cfg.batchsize=32
cfg.gpu='0'
cfg.workers=4


def main():
    global args
    # from config import cfg # use this if cfg class is on an external file called config.py
    parser = create_parser(cfg)
    args = parser.parse_args()
    print("args:", args)

if __name__ == '__main__':
    main()


