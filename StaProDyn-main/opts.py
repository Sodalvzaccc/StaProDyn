import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'dataset': [
            dict(name='--datasetName',
                 type=str,
                 default='sims',
                 help='mosi, mosei, sims or simsv2'),
            dict(name='--dataPath',
                 default="/root/autodl-tmp/3wd_drt/data/MSA Datasets/SIMS/Processed/unaligned_39.pkl",
                 type=str,
                 help=' '),
            dict(name='--seq_lens',
                 default=[39, 55, 400],
                 type=list,
                 help='the length of T, V, A modalities; sims: [39, 55, 400]; simsv2: []; mosi: []; mosei: []'),
            dict(name='--num_workers',
                 default=8,
                 type=int,
                 help=' '),
            dict(name='--train_mode',
                 default="regression",
                 type=str,
                 help=' '),
            dict(name='--full_data',
                 action='store_true',
                 help='use complete data without modality dropping'),
            dict(name='--drop_rate',
                 type=float,
                 default=0.6,
                 help='modality dropping probability'),
        ],

        'network': [
            dict(name='--fusion_layers', default=3, type=int),
            dict(name='--dropout', default=0.2, type=float),
            dict(name='--hidden_size', default=256, type=int),
            dict(name='--ffn_size', default=512, type=int),
            # Transformer related parameters
            dict(name='--attn_dropout', default=0.15, type=float, help='attention dropout'),
            dict(name='--attn_dropout_a', default=0.15, type=float, help='audio attention dropout'),
            dict(name='--attn_dropout_v', default=0.15, type=float, help='visual attention dropout'),
            dict(name='--relu_dropout', default=0.15, type=float, help='ReLU dropout'),
            dict(name='--embed_dropout', default=0.2, type=float, help='embedding dropout'),
            dict(name='--res_dropout', default=0.15, type=float, help='residual dropout'),
            dict(name='--out_dropout', default=0.1, type=float, help='output dropout'),
            dict(name='--nlevels', default=4, type=int, help='number of Transformer layers'),
            dict(name='--proj_dim', default=256, type=int, help='projection dimension'),
            dict(name='--num_heads', default=4, type=int, help='number of attention heads'),
            dict(name='--attn_mask', action='store_false', help='use attention mask'),
            dict(name='--prompt_dim', default=256, type=int),
            dict(name='--prompt_length', default=16, type=int),
            dict(name='--max_depth', nargs='+', type=int, default=5,
                 help='maximum depth values (can be list for different modules)'),
            dict(name='--output_droupout_prob', type=float, default=0.0),
            # SAF related parameters
            dict(name='--saf_mix_prob_max', type=float, default=0.1, help='max prob to use SAF for missing selection'),
            dict(name='--saf_warmup_epochs', type=int, default=8, help='epochs to warm up SAF prob'),
            dict(name='--saf_margin', type=float, default=0.15, help='min margin between weakest and second'),
            dict(name='--saf_balance', action='store_true', help='enable batch balance for SAF missing'),
            dict(name='--saf_balance_slack', type=int, default=1, help='extra allowance per class in batch balance'),
            dict(name='--saf_std_clip', type=float, default=1.0, help='clip std/var terms to avoid extremes')
        ],

        'common': [
            dict(name='--seed', default=1111, type=int),
            dict(name='--batch_size', default=32, type=int),
            dict(name='--lr', type=float, default=2e-5),
            dict(name='--weight_decay', type=float, default=1e-4),
            dict(name='--n_epochs', default=50, type=int, help='Number of total epochs to run'),
            dict(name='--log_path', default='./log/', type=str, help='Logger path for saving logs'),
            # Optimization related parameters
            dict(name='--clip', default=0.8, type=float, help='gradient clipping'),
            dict(name='--optim', default='Adam', type=str, help='optimizer type'),
            dict(name='--when', default=20, type=int, help='when to decay LR'),
            dict(name='--no_cuda', action='store_true', help='disable CUDA'),
            dict(name='--name', default=None, type=str, help='experiment name'),
            # NCE warmup
            dict(name='--nce_warmup_epochs', type=int, default=25, help='epochs to warm up NCE weight'),
            dict(name='--nce_max_weight', type=float, default=0.08, help='max NCE weight')
        ]
    }


    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)

    args = parser.parse_args()
    return args
