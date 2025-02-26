from argparse import ArgumentParser

class parser():
    def __init__():
        pass

    def __str__():
        pass

    def parseargs():
        parser = ArgumentParser(description='Meta system identification with transformers - IsaacGym')

        # Overall
        parser.add_argument('--train-folder', type=str, default="train", metavar='S',
                            help='train folder')
        parser.add_argument('--model-dir', type=str, default="out", metavar='S',
                            help='Saved model folder')
        parser.add_argument('--out-file', type=str, default="ckpt", metavar='S',
                            help='Saved model name')
        parser.add_argument('--in-file', type=str, default="ckpt", metavar='S',
                            help='Loaded model name (when resuming)')
        parser.add_argument('--init-from', type=str, default="scratch", metavar='S',
                            help='Init from (scratch|resume|pretrained)')
        parser.add_argument('--seed', type=int, default=42, metavar='N',
                            help='Seed for random number generation')
        parser.add_argument('--log-wandb', default=False,     # --log-wandb', action='store_true', default=False,
                            help='Live log')

        # Dataset
        parser.add_argument('--nx', type=int, default=7, metavar='N',
                            help='model order (default: 5)')
        parser.add_argument('--nu', type=int, default=7, metavar='N',
                            help='model order (default: 5)')
        parser.add_argument('--ny', type=int, default=14, metavar='N',
                            help='model order (default: 5)')

        parser.add_argument('--seq-len-ctx', type=int, default=400, metavar='N',
                            help='sequence length (default: 300)')
        parser.add_argument('--seq-len-new', type=int, default=400, metavar='N',
                            help='sequence length (default: 300)')
        parser.add_argument('--mag_range', type=tuple, default=(0.5, 0.97), metavar='N',
                            help='sequence length (default: 600)')
        parser.add_argument('--phase_range', type=tuple, default=(0.0, math.pi/2), metavar='N',
                            help='sequence length (default: 600)')
        parser.add_argument('--fixed-system', action='store_true', default=False,
                            help='If True, keep the same model all the times')

        # Model
        parser.add_argument('--n-layer', type=int, default=12, metavar='N',
                            help='number of iterations (default: 1M)')
        parser.add_argument('--n-head', type=int, default=4, metavar='N',
                            help='number of iterations (default: 1M)')
        parser.add_argument('--n-embd', type=int, default=128, metavar='N',
                            help='number of iterations (default: 1M)')
        parser.add_argument('--dropout', type=float, default=0.0, metavar='LR',
                            help='learning rate (default: 1e-4)')
        parser.add_argument('--bias', action='store_true', default=True,
                            help='bias in model')

        # Training
        parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                            help='batch size (default:32)')
        parser.add_argument('--max-iters', type=int, default=1_000_000, metavar='N',
                            help='number of iterations (default: 1M)')
        parser.add_argument('--warmup-iters', type=int, default=10_000, metavar='N',
                            help='number of iterations (default: 1000)')
        parser.add_argument('--lr', type=float, default=6e-4, metavar='LR',
                            help='learning rate (default: 1e-4)')
        parser.add_argument('--weight-decay', type=float, default=0.0, metavar='D',
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--eval-interval', type=int, default=100, metavar='N',
                            help='batch size (default:32)')
        parser.add_argument('--eval-iters', type=int, default=50, metavar='N',
                            help='batch size (default:32)')
        parser.add_argument('--fixed-lr', action='store_true', default=False,
                            help='disables CUDA training')

        # Compute
        parser.add_argument('--threads', type=int, default=10,
                            help='number of CPU threads (defa400ult: 10)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--cuda-device', type=str, default="cuda:0", metavar='S',
                            help='cuda device (default: "cuda:0")')
        parser.add_argument('--compile', action='store_true', default=False,
                            help='disables CUDA training')

        # ------------------------------------ Added ---------------------------------------------

        parser.add_argument('--custom-dataset', action='store_true', default=True,
                        help='Dataset from Isaac or simple sys')
        parser.add_argument('--manuel_pc',default=True,   
                        help='if you are Training on lab --> False')
        parser.add_argument('--context', type=float, default=.2,
                    help='How much of the timesteps is context')
        parser.add_argument('--loss-function', type=str, default='MSE',
                    help="Loss function: 'MAE'-'MSE'-'Huber'")
        parser.add_argument('--iter-log', type=int, default=100,
                            help='Iteration every which logs wandb')

        # ------------------------------------ Edited ------------------------------------
        parser.add_argument('--beta1', type=float, default=.9,
                    help="Loss function: 'MAE'-'MSE'-'Huber'")
        parser.add_argument('--beta2', type=float, default=.95,
                    help="Loss function: 'MAE'-'MSE'-'Huber'")
        
        return parser