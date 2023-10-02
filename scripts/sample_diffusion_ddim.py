import argparse, os, gc, glob, datetime, yaml
import logging
import math

import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.cuda import amp
from pytorch_lightning import seed_everything

from ddim.models.diffusion import Model
from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from ddim.functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu

from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples

logger = logging.getLogger(__name__)


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        config.split_shortcut = self.args.split
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float()
        self.betas = self.betas.to(self.device)
        betas = self.betas
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        model = Model(self.config)

        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError
        ckpt = get_ckpt_path(f"ema_{name}")
        logger.info("Loading checkpoint {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        
        model.to(self.device)
        model.eval()
        assert(self.args.cond == False)
        if self.args.ptq:
            if self.args.quant_mode == 'qdiff':
                wq_params = {'n_bits': args.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
                aq_params = {'n_bits': args.act_bit, 'symmetric': args.a_sym, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': args.quant_act}
                if self.args.resume:
                    logger.info('Load with min-max quick initialization')
                    wq_params['scale_method'] = 'max'
                    aq_params['scale_method'] = 'max'
                if self.args.resume_w:
                    wq_params['scale_method'] = 'max'
                qnn = QuantModel(
                    model=model, weight_quant_params=wq_params, act_quant_params=aq_params, 
                    sm_abit=self.args.sm_abit)
                qnn.to(self.device)
                qnn.eval()

                if self.args.resume:
                    image_size = self.config.data.image_size
                    channels = self.config.data.channels
                    cali_data = (torch.randn(1, channels, image_size, image_size), torch.randint(0, 1000, (1,)))
                    resume_cali_model(qnn, args.cali_ckpt, cali_data, args.quant_act, "qdiff", cond=False)
                else:
                    logger.info(f"Sampling data from {self.args.cali_st} timesteps for calibration")
                    sample_data = torch.load(self.args.cali_data_path)
                    cali_data = get_train_samples(self.args, sample_data, custom_steps=0)
                    del(sample_data)
                    gc.collect()
                    logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape}")

                    cali_xs, cali_ts = cali_data
                    if self.args.resume_w:
                        resume_cali_model(qnn, self.args.cali_ckpt, cali_data, False, cond=False)
                    else:
                        logger.info("Initializing weight quantization parameters")
                        qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
                        _ = qnn(cali_xs[:8].cuda(), cali_ts[:8].cuda())
                        logger.info("Initializing has done!")

                    # Kwargs for weight rounding calibration
                    kwargs = dict(cali_data=cali_data, batch_size=self.args.cali_batch_size, 
                                iters=self.args.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                                warmup=0.2, act_quant=False, opt_mode='mse')

                    def recon_model(model):
                        """
                        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
                        """
                        for name, module in model.named_children():
                            logger.info(f"{name} {isinstance(module, BaseQuantBlock)}")
                            if isinstance(module, QuantModule):
                                if module.ignore_reconstruction is True:
                                    logger.info('Ignore reconstruction of layer {}'.format(name))
                                    continue
                                else:
                                    logger.info('Reconstruction for layer {}'.format(name))
                                    layer_reconstruction(qnn, module, **kwargs)
                            elif isinstance(module, BaseQuantBlock):
                                if module.ignore_reconstruction is True:
                                    logger.info('Ignore reconstruction of block {}'.format(name))
                                    continue
                                else:
                                    logger.info('Reconstruction for block {}'.format(name))
                                    block_reconstruction(qnn, module, **kwargs)
                            else:
                                recon_model(module)

                    if not self.args.resume_w:
                        logger.info("Doing weight calibration")
                        recon_model(qnn)
                        qnn.set_quant_state(weight_quant=True, act_quant=False)
                    if self.args.quant_act:
                        logger.info("UNet model")
                        logger.info(model)                    
                        logger.info("Doing activation calibration")   
                        # Initialize activation quantization parameters
                        qnn.set_quant_state(True, True)
                        with torch.no_grad():
                            inds = np.random.choice(cali_xs.shape[0], 64, replace=False)
                            # _ = qnn(cali_xs[:64].cuda(), cali_ts[:64].cuda())
                            _ = qnn(cali_xs[inds].cuda(), cali_ts[inds].cuda())
                        
                            if self.args.running_stat:
                                logger.info('Running stat for activation quantization')
                                qnn.set_running_stat(True)
                                for i in range(int(cali_xs.size(0) / 64)):
                                    _ = qnn(
                                        (cali_xs[i * 64:(i + 1) * 64].to(self.device), 
                                        cali_ts[i * 64:(i + 1) * 64].to(self.device)))
                                qnn.set_running_stat(False)
                        
                        kwargs = dict(
                            cali_data=cali_data, iters=self.args.cali_iters_a, act_quant=True, 
                            opt_mode='mse', lr=self.args.cali_lr, p=self.args.cali_p)   
                        recon_model(qnn)
                        qnn.set_quant_state(weight_quant=True, act_quant=True)   

                    logger.info("Saving calibrated quantized UNet model")
                    for m in qnn.model.modules():
                        if isinstance(m, AdaRoundQuantizer):
                            m.zero_point = nn.Parameter(m.zero_point)
                            m.delta = nn.Parameter(m.delta)
                        elif isinstance(m, UniformAffineQuantizer) and self.args.quant_act:
                            if m.zero_point is not None:
                                if not torch.is_tensor(m.zero_point):
                                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                                else:
                                    m.zero_point = nn.Parameter(m.zero_point)
                    torch.save(qnn.state_dict(), os.path.join(self.args.logdir, "ckpt.pth"))

                model = qnn

        model.to(self.device)
        if self.args.verbose:
            logger.info("quantized model")
            logger.info(model)

        model.eval()

        self.sample_fid(model)
        

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        logger.info(f"starting from image {img_id}")
        total_n_samples = self.args.max_images
        n_rounds = math.ceil((total_n_samples - img_id) / config.sampling.batch_size)

        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        with torch.no_grad():
            for i in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                with amp.autocast(enabled=False):
                    x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                if img_id + x.shape[0] > self.args.max_images:
                    assert(i == n_rounds - 1)
                    n = self.args.max_images - img_id
                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from ddim.functions.denoising import generalized_steps

            betas = self.betas
            xs = generalized_steps(
                x, seq, model, betas, eta=self.args.eta, args=self.args)
            x = xs
        elif self.args.sample_type == "dpm_solver":
            logger.info(f"use dpm-solver with {self.args.timesteps} steps")
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
            model_fn = model_wrapper(
                model,
                noise_schedule,
                model_type="noise"
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
            return dpm_solver.sample(
                x,
                steps=self.args.timesteps,
                order=3,
                skip_type="time_uniform",
                method="singlestep",
            )
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--quant_mode", type=str, default="qdiff", 
        choices=["qdiff"], 
        help="quantization mode to use"
    )
    parser.add_argument(
        "--max_images", type=int, default=50000, help="number of images to sample"
    )

    # qdiff specific configs
    parser.add_argument(
        "--cali_st", type=int, default=1, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=32, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=20000, 
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int, 
        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, 
        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, 
        help='L_p norm minimization for LSQ')
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cali_data_path", type=str, default="sd_coco_sample1024_allst.pt",
        help="calibration dataset name"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--resume_w", action="store_true",
        help="resume the calibrated qdiff model weights only"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--a_sym", action="store_true",
        help="act quantizers use symmetric quantization"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument("--split", action="store_true",
        help="split shortcut connection into two parts"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    return parser


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = get_parser()
    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # fix random seed
    seed_everything(args.seed)

    # setup logger
    logdir = os.path.join(args.logdir, "samples", now)
    os.makedirs(logdir)
    args.logdir = logdir
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(75 * "=")
    logger.info(f"Host {os.uname()[1]}")
    logger.info("logging to:")
    imglogdir = os.path.join(logdir, "img")
    args.image_folder = imglogdir

    os.makedirs(imglogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    runner = Diffusion(args, config)
    runner.sample()