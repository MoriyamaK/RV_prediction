
import os
import sys
import tempfile
import getpass
import random
import string
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from torch import nn
from torch.optim import SGD, Adam, Adagrad, AdamW
from torch.optim.lr_scheduler import OneCycleLR
import logging

def config_logger(level):
    from rich.logging import RichHandler
    from rich.console import Console
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=level, format=FORMAT, datefmt="[%X]",
        handlers=[RichHandler(console=Console(width=120))]
    )
    #logging.getLogger().setLevel(logging.DEBUG)

from module.flow import (
    IdentityFlow,
    NeuralODEFlow,
    StackedFlow,
    WallaceFlow,
    LogFlow,
)
from module.flow import (
    ExtendedNeuralODEFlow,
    ExtendedStackedFlow,
)
from module import (
    GCFP1D,
    NamedEmbedding,
)
from module.utils import (
    build_gcfp,
    save_checkpoint,
    load_from_checkpoint,
)
from module_utils import (
    set_seed,
    make_sequential_dataloader,
)
from module.utils.evaluate import (
    evaluate_NLL_GCFP1D,
    collate_NLL,
    evaluate_SE_GCFP1D,
    collate_SE,
    evaluate_AE_GCFP1D,
    collate_AE,
)


EXCEPTIONAL_LOGDET = 10.0


class DummyOptimizer:
    def step(self): pass
    def zero_grad(self): pass


class DummyScheduler:
    def step(self): pass
    def get_last_lr(self): return [0]


def parse_arguments():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # ----- experiment setting -----
    parser.add_argument('--ret_hist_path', type=str)
    parser.add_argument('--train_start_step', type=int, default=0)
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--eval_start_step', type=int, default=None)
    parser.add_argument('--eval_steps', type=int)
    parser.add_argument('--test_start_step', type=int, default=None)
    parser.add_argument('--test_steps', type=int, default=9999, 
                        help='set large to test on all remaining steps')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='model is evaluated every 5 iterations and the snapshot is saved.')
    parser.add_argument('--predmodel', type=str, default='garch11',
                        choices=['constvar', 'garch11', 'gjr11', 'tarch11', 'fastnsvm', 'fastnsvm_zeromean', 'taylor', 'har'],
                        help='volatility prediction model')
    parser.add_argument('--predmodel_dist', type=str, choices=['normal', 'ged', 't', 'skewt'], default='normal',
                        help='Presumed distribution of the volatility prediction model. ')
    parser.add_argument('--flow', type=str, choices=['identity', 'node', 'wallace', 'log', 'yeojohnson',  'boxcox', 'tanh'], default='node',
                        help='Type of the normalizing flow. Identity normalizing flow is f(x)=x .')
    parser.add_argument('--dim', type=int, default=1,
                        help='Dimension of time series. Only `dim=1` is supported now.')
    parser.add_argument('--lag', type=int, default=0)

    # ----- incorporating asset-wise shape parameter varphi -----
    parser.add_argument('--phi_dim', type=int, default=1, help='Only `phi_dim=0 or 1` is supported for now.')

    # ----- NODE parameters -----
    parser.add_argument('--d_hidden', type=str, default='4,4')
    parser.add_argument('--divergence_fn', type=str, default='approximate', choices=['approximate', 'brute_force'],
                        help='The method to compute log determinant of NODE at training stage, '
                             'either `approximate` (inexact) or `brute_force` (inexact). '
                             'For evaluation stage, this value is fixed to `brute_force` by the '
                             'program, and the results are always exact.')
    parser.add_argument('--time_length', type=float, default=5.0)
    parser.add_argument('--train_T', action='store_true')
    parser.add_argument('--layer_type', type=str, default='ignore')
    parser.add_argument('--nonlinearity', type=str, default='swish')
    parser.add_argument('--n_blocks', type=int, default=1)
    parser.add_argument('--has_batchnorm', action='store_true')
    parser.add_argument('--batchnorm_lag', type=int, default=0)
    parser.add_argument('--ode_solver', type=str, default='dopri5')
    parser.add_argument('--atol', type=float, default=1e-5, help='Error tolerance')
    parser.add_argument('--rtol', type=float, default=1e-5, help='Error tolerance')
    parser.add_argument('--step_size', type=float, default=None)
    parser.add_argument('--test_ode_solver', type=str, default=None)
    parser.add_argument('--test_atol', type=float, default=1e-5)
    parser.add_argument('--test_rtol', type=float, default=1e-5)
    parser.add_argument('--no_rademacher', action='store_true')
    parser.add_argument('--residual', action='store_true')

    # ----- optimize -----
    parser.add_argument('--n_iters', type=int, default=200)
    parser.add_argument('--sz_batch', type=int, default=9999,
                        help='set it large to put all assets in one batch, or make it small to reduce memory usage')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=10.0)

    # ----- miscellaneous -----
    parser.add_argument('--seed', type=int, default=2222)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--cuda_visible_devices', type=str, default=None)
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--summary', action='store_true', help='print only summaries')
    parser.add_argument('--summary_path', type=str, default=None)


    # ----- wandb -----
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    config_logger(args.log_level.upper())

    if args.eval_start_step is None:
        args.eval_start_step = args.train_start_step + args.train_steps
    if args.test_start_step is None:
        args.test_start_step = args.eval_start_step + args.eval_steps

    return args


def train():
    args = parse_arguments()
    #logging.getLogger().setLevel(logging.DEBUG)
    # --------------- arguments checking ------------------
    if args.summary:
        logging.getLogger().setLevel(logging.INFO)


    logging.debug("Args:")
    for k, v in sorted(args.__dict__.items(), key=lambda x: x[0]):
        logging.debug(f'{str(k):<20} : {str(v):<}')

    if args.log_path is not None:
        f = open(args.log_path, 'w')
        sys.stdout = f
        sys.stderr = f

    args.use_cuda = not args.cpu
    args.use_cuda = torch.cuda.is_available() and args.use_cuda
    if args.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    args.device = 'cuda' if args.use_cuda else 'cpu'

    exp_name = ''.join(random.choices(
        string.ascii_uppercase + string.digits, k=8))
    if args.save_dir is None:
        args.save_dir = f'{tempfile.gettempdir()}/{getpass.getuser()}_gcfp/'
    args.save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    if args.use_wandb:
        import wandb
        from wandb_setting import config
        config()
        wandb.init(name=exp_name, config=args.__dict__)
        wandb.save(f'{args.save_dir}/*', args.save_dir)
        
    
        
    """ data preparation """
    ret = pd.read_csv(args.ret_hist_path, index_col=0)
    ret = ret.astype(np.float32)
    asset_symbs = ret.columns
    n_assets = len(asset_symbs)
    ret = ret.apply(lambda x: (x-x.mean())/x.std(), axis=0)

    assert args.train_start_step + args.train_steps <= len(ret)
    assert args.eval_start_step + args.eval_steps <= len(ret)
    args.test_steps = min(args.test_steps, len(ret) - args.test_start_step)

    args.sz_batch = min(args.sz_batch, ret.shape[1])
    logging.debug(f'test_steps = {args.test_steps}')
    logging.debug(f'sz_batch = {args.sz_batch}')

    train_dataloader = make_sequential_dataloader(
        ret.iloc[args.train_start_step:
                 args.train_start_step+args.train_steps],
        dim=args.dim, sz_batch=args.sz_batch, device=args.device,
    )


    # --------------------- model & optimizer construction --------------------

    lag = args.lag

    phis = NamedEmbedding(n_assets, asset_symbs, args.phi_dim)
    if args.phi_dim == 1:
        # if phi_dim is one, initialize phis with asset kurtoses
        kurtosis = ret.var()
        phis.weight.data = torch.tensor(kurtosis, dtype=torch.float32, device=args.device).view(-1, 1)**0.25


    set_seed(args.seed)

    gcfp: GCFP1D = build_gcfp(args)
    gcfp = gcfp.to(args.device)
    logging.debug(gcfp)
    
    gcfp_parameters = list(gcfp.parameters())
    if args.phi_dim > 0:
        parameters = gcfp_parameters + list(phis.parameters())
    else:
        parameters = gcfp_parameters
    logging.debug(f'gcfp parameters: {gcfp_parameters}')
    if len(gcfp_parameters):
        args.optmizer = args.optimizer.lower()
        if args.optimizer == 'adam':
            optimizer = Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
            scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.n_iters+5)
        elif args.optimizer == 'adamw':
            optimizer = AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
            scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.n_iters+5)
        elif args.optimizer == 'sgd':
            optimizer = SGD(parameters, lr=args.lr, weight_decay=args.weight_decay)
            scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.n_iters+5)
        elif args.optimizer == 'adagrad':
            optimizer = Adagrad(parameters, lr=args.lr, weight_decay=args.weight_decay)
            scheduler = DummyScheduler()
    else:
        optimizer = DummyOptimizer()
        scheduler = DummyScheduler()

    # ------------------------------ training -------------------------------

    tbar = range(0, 1 + args.n_iters)
    best_valid_nll_x_mean = float('inf')
    best_rmse_x = float('inf')
    best_iter = None
    best_savedir = f"{args.save_dir}/valid-best"

    n_batches = len(train_dataloader)

    datait = iter(train_dataloader)
    for i in tbar:

        optimizer.zero_grad()

        try:
            x, symbs_list = next(datait)
        except StopIteration:
            datait = iter(train_dataloader)
            x, symbs_list = next(datait)
        x = x.to(args.device)
    
        phi = phis(symbs_list)
        # phi: (sz_batch, dim=1, phi_dim)
        phi = phi.view(args.sz_batch, args.dim * args.phi_dim)
        # phi: (sz_batch, dim * phi_dim)
        

        x_train = x
        x_train = torch.transpose(x_train, 0, 1).contiguous()
        # x_train: (sz_batch, train_steps, dim)
        gcfp.train()
        
        """ nll in the base process """
        latent_nll, logdet, aux_losses = gcfp.nll(
            x_train, phi, last_obs=-1, return_auxiliary_losses=True,)
        mse = 0
        
        nll = (latent_nll[:, lag:] + logdet[:, lag:]).mean()
        mean_logdet = logdet.data.mean().abs().item() 
        
        if mean_logdet > EXCEPTIONAL_LOGDET:
            logging.warn(f"Unexpected large logdet {mean_logdet}.")
            if best_savedir is not None:
                ckp = load_from_checkpoint(best_savedir, map_location=args.device)
                gcfp, phis = ckp['gcfp'], ckp['phis']
                logging.warn(f'Loaded best snapshot from {best_savedir}')
            else:
                logging.error('No saved model snapshots found. Program ended.')
                sys.exit()
            continue
    
        """ sum up all losses """
        
        loss = sum([nll])# + [val for key, val in aux_losses.items()])
        if len(gcfp_parameters):

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            scheduler.step()
        else:
            grad_norm = 0

        flow_volratio = logdet.mean().exp().item()  # volume ratio caused by `flow`
        logging.debug('phis: ' + str(phis.weight.data.squeeze()))
        logging.info(
            f"Iter: {i:0>4d}: "
            f"lr={scheduler.get_last_lr()[0]:.3g}, "
            f"grad={grad_norm:.3g}, "
            f"basenll={nll.item():.3g} (_nll={latent_nll[:, lag:].mean():.3g}, logdet={logdet[:, lag:].mean():.3g}), "
            f"totloss={loss.item():.3g}, "
            f"flowratio={flow_volratio:.3g}, "
        )
        if args.use_wandb:
            wandb.log({
                'train/total_loss': loss.item(),
                'train/flow_volratio': flow_volratio,
                'train/train_mse': mse,
            }, step=i, commit=False)

        """ evaluate by autoregressive forecasting """
        if i % args.eval_interval == 0:
            
            metrics = evaluate_and_log(ret, gcfp, phis,
                             eval_start_step=args.eval_start_step,
                             eval_steps=args.eval_steps,
                             prefix='valid', iter=i, args=args)
            
            nll_x_mean = metrics['nll_x_mean']
            se_x = metrics['se_x']
            rmse_x = ((se_x.mean())**0.5).mean()
            """ update best && save model """
            if nll_x_mean < best_valid_nll_x_mean:
                best_valid_nll_x_mean = nll_x_mean
                best_rmse_x = rmse_x
                best_iter = i
                save_checkpoint(best_savedir, gcfp, phis, args.__dict__)
                if args.use_wandb:
                    wandb.save(f'{best_savedir}/*', base_path=args.save_dir)


    # ----------------------- evaluating the best model in eval on test set -------------------------
    logging.debug('\n\n\n')
    logging.info('----------------------------------------------------------------')
    logging.info('Testing on the test set...')
    logging.info('----------------------------------------------------------------')
    if best_iter is None:
        logging.debug("No feasible model found... Skip test.")
        return
    logging.info(f"Best evaluation iteration = {best_iter}." 
                 f"Validset RMSE = {best_rmse_x:.5f}."
                 f"Validset NLL = {best_valid_nll_x_mean:.5f}.")
    ckp = load_from_checkpoint(best_savedir, map_location=args.device)
    gcfp, phis = ckp['gcfp'], ckp['phis']
    logging.debug(f"Model loaded from {best_savedir}")
    if args.use_wandb:
        wandb.summary['best_iter'] = best_iter
        wandb.summary['best_valid_nll_x_mean'] = best_valid_nll_x_mean
        wandb.summary['best_rmse_x'] = best_rmse_x

    metrics = evaluate_and_log(
        ret, gcfp, phis,
        eval_start_step=args.test_start_step,
        eval_steps=args.test_steps,
        prefix='test', iter=None,
        args=args)
    
    logging.info(metrics['se_x'].mean()**0.5)
    logging.debug("Args:")
    for k, v in sorted(args.__dict__.items(), key=lambda x: x[0]):
        logging.debug(f'{str(k):<20} : {str(v):<}')

    if not args.use_wandb:
        return {
            'test/se_x/base': metrics['se_x'],
            'test/mse_x/base': metrics['mse_x'],
        }


def evaluate_and_log(ret, gcfp, phis,
                     eval_start_step, eval_steps, prefix, iter, args):
    if args.use_wandb: import wandb
    
    results = evaluate(
        ret, gcfp, phis,
        eval_start_step=eval_start_step,
        eval_steps=eval_steps,
        eval_hooks={
            'NLL': (
                (evaluate_NLL_GCFP1D, {}),
                (collate_NLL, {})
                ), 
            'SE': (
                (evaluate_SE_GCFP1D, {}),
                (collate_SE, {})
            ),
            'AE': (
                (evaluate_AE_GCFP1D, {}),
                (collate_AE, {})
            ),
        },
        args=args,
    )
    
    nlls_latent = results['NLL']
    nll_mean_latent, nll_std_latent = nlls_latent.mean(axis=0).mean(), nlls_latent.mean(axis=0).std()
    
    logging.info(
        f"{prefix} NLL mean: {nll_mean_latent:g}.")
    nlls_latent.to_csv(f'{args.save_dir}/{prefix}.nll.csv')
    logging.debug(f'\n{nlls_latent.mean()}')
    if args.use_wandb:
        wandb.log({
            f'{prefix}/nll_x_mean/base': nll_mean_latent,
            f'{prefix}/nll_x_std/base': nll_std_latent,
            f'{prefix}/nll_x/base': wandb.Table(dataframe=nlls_latent),
        }, step=iter, commit=False)
    
    

    # ---- test: MSE
    ses = results['SE']
    #ses_latent[ses_latent > 100] = 100 for abnormal value
    mse, se_std = ses.mean(axis=0).mean(), ses.mean(axis=0).std()
    logging.info(
        f"{prefix} RMSE: {(ses.mean()**0.5).mean():g}.")
    ses.to_csv(f'{args.save_dir}/{prefix}.se.csv')
    logging.debug(f'\n{ses.mean()}')
    if args.use_wandb:
        wandb.log({
            f'{prefix}/se_x_mean/base': mse,
            f'{prefix}/se_x_std/base': se_std,
            f'{prefix}/se_x/base': wandb.Table(dataframe=ses),
        }, step=iter, commit=False)
    
    # ---- test: MAE
    aes = results['AE']
    aes[aes > 100] = 100
    mae, ae_std = aes.mean(axis=0).mean(), aes.mean(axis=0).std()
    logging.info(
        f"{prefix} MAE: {(aes.mean()).mean():g}.")
    aes.to_csv(f'{args.save_dir}/{prefix}.ae.csv')
    logging.debug(f'\n{aes.mean()}')
    if args.use_wandb:
        wandb.log({
            f'{prefix}/ae_x_mean/base': mae,
            f'{prefix}/ae_x_std/base': ae_std,
            f'{prefix}/ae_x/base': wandb.Table(dataframe=aes),
        }, step=iter, commit=False)


    return {
        'nll_x_mean': nll_mean_latent,
        'nll_x': nlls_latent,
        'mse_x': mse,
        'se_x': ses, 
        'mae' : mae, 
        'ae_x': aes,
    }



def evaluate(ret,
             gcfp: GCFP1D,
             phis,
             eval_start_step,
             eval_steps,
             eval_hooks: dict,
             args,
             ):
    
    logging.info(f'Evaluating ... '
                 f'eval_start_step={eval_start_step}, eval_steps={eval_steps}')
    
    total_steps = eval_start_step + eval_steps

    #ret = ret.iloc[:total_steps]
    ret = ret.iloc[:total_steps]

    dataloader = make_sequential_dataloader(
        ret, dim=args.dim, sz_batch=args.sz_batch, device=args.device)
    
    buffer = defaultdict(list)

    for i, (r, symblists) in enumerate(dataloader):
        r = r.to(args.device)
        # r: (eval_start_step + eval_steps, sz_batch, dim)
        phi = phis(symblists)
        # phi: (sz_batch, dim=1, phi_dim)
        phi = phi.view(args.sz_batch, -1)
        # phi: (sz_batch, dim * phi_dim)

        gcfp.eval()    

        with torch.no_grad():
            x = torch.transpose(r, 0, 1).contiguous()
            #x = x[:, :total_steps, :].contiguous()
            x = x[:, args.train_start_step:total_steps, :].contiguous()
            # x: (sz_batch, tend, dim)

            last_obs = eval_start_step - args.train_start_step
            # ------------------- out-of-sample nll --------------------
            for hname, ((func, kwargs), _) in eval_hooks.items():
                batch_res = func(gcfp, phi, x, last_obs,
                                **kwargs)
                buffer[hname].append(batch_res)
    
    results = {}
    #use a merger like 
    symbs = ret.columns
    for hname, (_, merge_hook) in eval_hooks.items():
        if isinstance(merge_hook, tuple):
            merger, kwargs = merge_hook
            results[hname] = merger(
                buffer[hname], symbs=symbs, **kwargs)
        elif isinstance(merge_hook, dict):
            for subhname, (merger, kwargs) in merge_hook.items():
                results[f'{hname}/{subhname}'] = merger(
                    buffer[hname], symbs=ret.columns, **kwargs)
        else:
            raise ValueError
        
    return results
                

if __name__ == "__main__":
    train()
