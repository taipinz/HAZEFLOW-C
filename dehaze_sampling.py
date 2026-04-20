import os
# import sys
join = os.path.join
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import torch
from models import utils as mutils
from models.ema import ExponentialMovingAverage
# Keep the import below for registering all model definitions
from models import hazeflow, NAFNet_arch
import reflow.datasets as datasets
from reflow.utils import restore_checkpoint, seed_everywhere, save_image_batch
from reflow import RectifiedFlow
from reflow import losses as losses
from reflow import sampling as sampling
import torch.nn.functional as F

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)


def create_iqa_metrics():
    import pyiqa

    return {
        'brisque': pyiqa.create_metric('brisque'),
        'nima': pyiqa.create_metric('nima', base_model_name='vgg16', train_dataset='ava', num_classes=10),
        'musiq': pyiqa.create_metric('musiq'),
        'paq2piq': pyiqa.create_metric('paq2piq'),
    }

def main(argv):
    config = FLAGS.config

    ### basic info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    print(f'Using device: {device}; version: {str(torch.version.cuda)}')

    ### set random seed everywhere
    seed_everywhere(config.seed)

    ### create model & optimizer
    # Initialize model.
    score_model = mutils.create_model(config) 
    score_model.to(device)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Load pre-trained model if specified
    flow = RectifiedFlow(model=score_model, ema_model=ema, cfg=config)
    flow.model.eval()

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    if config.data.dataset == 'rtts':
        dataloader = datasets.get_rtts_dataloader(config) ## RTTS
    elif config.data.dataset == 'urhi':
        dataloader = datasets.get_test_dataloader(config,dataset_name='urhi') ## URHI
    elif config.data.dataset == 'etc':
        dataloader = datasets.get_real_dataloader(config) ## etc
    elif config.data.dataset == 'custom':
        dataloader = datasets.get_test_dataloader(config,dataset_name='custom') ## custom
    
    # reset random seed for Initial noise
    seed_everywhere(config.seed)
    # Initial noise fixed
    
    class_labels = None

    ckpt = config.sampling.ckpt
    state = restore_checkpoint(ckpt, state, device=config.device)
    
    ema.copy_to(score_model.parameters())
    
    sampling_fn_n1 = sampling.get_flow_sampler(
        flow,
        device=device,
        use_ode_sampler=config.sampling.use_ode_sampler,
    )
    
    sample_dir = os.path.join(config.work_dir, config.expr) # '/data/RF/samples/URHI_20step/'
    compute_iqa = config.eval.compute_iqa
    metrics = create_iqa_metrics() if compute_iqa else None
    metric_sums = {metric_name: 0.0 for metric_name in metrics} if metrics is not None else {}
    count = 0
    for idx, batch in enumerate(dataloader): 
        seed_everywhere(config.seed)
        
        hazy = batch['hazy'].to(device)
        dcp = batch['dcp'].to(device)
        name = batch['name'][0]
        if config.sampling.use_ode_sampler == 'depth_aware_euler':
            depth = batch['depth'].to(device) if batch['depth'] is not None else None 
        else: 
            depth = None
            
        img_multiple_of = 128
        height, width = hazy.shape[2], hazy.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        z0 = F.pad(hazy, (0, padw, 0, padh), 'reflect')                    
        t = F.pad(dcp, (0, padw, 0, padh), 'reflect')
        if depth is not None:
            depth = F.pad(depth, (0, padw, 0, padh), 'reflect')
        

        if H * W > 1600*1600:
            z0 = torch.nn.UpsamplingBilinear2d((H//2, W//2))(z0)
            sample_n1, pred_t = sampling_fn_n1(score_model, hazy=z0, init_t = t, label=class_labels)
            sample_n1 = torch.nn.UpsamplingBilinear2d((H, W))(sample_n1)
            sample_n1 = sample_n1[:,:,:height, :width] 
            pred_t = pred_t[:,:,:height, :width]   
            t = t[:,:,:height, :width]     
            
        else:
            sample_n1, pred_t = sampling_fn_n1(score_model, hazy=z0, init_t = t, label=class_labels)
            sample_n1 = sample_n1[:,:,:height, :width]  
            pred_t = pred_t[:,:,:height, :width]   
            t = t[:,:,:height, :width]   
        ema.restore(score_model.parameters())
                
        this_sample_dir = os.path.join(sample_dir)
        os.makedirs(this_sample_dir, exist_ok=True)
        if config.data.dataset == 'etc':
            os.makedirs(os.path.join(this_sample_dir,name.split('/')[0]), exist_ok=True)

        save_image_batch(sample_n1, config.data.image_size, this_sample_dir, log_name=name)  
        if metrics is not None:
            image = sample_n1
            for metric_name, metric_fn in metrics.items():
                metric_sums[metric_name] += float(metric_fn(image))
            count += 1

        # this_pred_t_dir = os.path.join(sample_dir,'pred_t')
        # os.makedirs(this_pred_t_dir, exist_ok=True)
        # save_image_batch(pred_t, config.data.image_size, this_pred_t_dir, log_name=name) 

        # this_dcp_t_dir = os.path.join(sample_dir,'dcp_t')
        # os.makedirs(this_dcp_t_dir, exist_ok=True)
        # save_image_batch(t, config.data.image_size, this_dcp_t_dir, log_name=name)       
    if metrics is not None and count > 0:
        metric_lines = []
        for metric_name, metric_sum in metric_sums.items():
            metric_value = metric_sum / count
            metric_lines.append(f'Average {metric_name.upper()}: {metric_value:.4f}')
        metrics_path = os.path.join(sample_dir, 'iqa_metrics.txt')
        with open(metrics_path, 'w') as metric_file:
            metric_file.write('\n'.join(metric_lines) + '\n')
        for line in metric_lines:
            print(line)
        print(f'Saved IQA summary to {metrics_path}')
        

if __name__ == "__main__":
    app.run(main)
