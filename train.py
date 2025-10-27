import os
import torch
import time
from tqdm import tqdm
from opts import parse_opts
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, ConfigLogging, save_print_results, calculate_u_test
from models.OverallModal import build_model
from core.metric import MetricsTop


opt = parse_opts()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(parse_args):
    opt = parse_args

    log_path = os.path.join(opt.log_path, opt.datasetName.upper())
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, time.strftime('%Y-%m-%d-%H:%M:%S' + '.log'))
    logger = ConfigLogging(log_file)
    logger.info(opt)    # 保存当前模型参数

    setup_seed(opt.seed)
    dataLoader = MMDataLoader(opt)
    for data in dataLoader['train']:
        # 获取实际的特征维度
        opt.orig_d_l = data['text'].shape[-1]  # 文本特征维度
        opt.orig_d_v = data['vision'].shape[-1]  # 视频特征维度
        opt.orig_d_a = data['audio'].shape[-1]  # 音频特征维度
        # [MODIFIED] 获取到维度后即可跳出
        break

    model = build_model(opt).to(device)
    model.preprocess_model(pretrain_path={
       'T': "./pretrainedModel/KnowledgeInjectPretraining/MOSI/MOSI_T_MAE-0.7465000152587891_Corr-0.7707.pth",
       'V': "./pretrainedModel/KnowledgeInjectPretraining/MOSI/MOSI_V_MAE-1.6024999618530273_Corr-0.0176.pth",
       'A': "./pretrainedModel/KnowledgeInjectPretraining/MOSI/MOSI_A_MAE-1.4140000343322754_Corr-0.2249.pth"
    })      # 加载预训练权重并冻结参数

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )

    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)
    scheduler_warmup = get_scheduler(optimizer, opt.n_epochs)

    for epoch in range(1, opt.n_epochs+1):
        train_results = train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)
        valid_results = evaluate(model, dataLoader['valid'], optimizer, loss_fn, epoch, metrics)
        test_results = test(model, dataLoader['test'], optimizer, loss_fn, epoch, metrics)
        save_print_results(opt, logger, train_results, valid_results, test_results)
        scheduler_warmup.step()


def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    train_pbar = tqdm(train_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.train()
    for data in train_pbar:
        inputs = {
            'V': data['vision'].to(device),
            'A': data['audio'].to(device),
            'T': data['text'].to(device),
            'mask': {
                'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device).to(torch.bool),
                'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device).to(torch.bool),
                'T': []
            }
        }
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        copy_label = label.clone().detach()
        batchsize = inputs['V'].shape[0]

        missing_mode = data['missing_mode'].to(device) if 'missing_mode' in data else torch.full((inputs['V'].size(0),), 6, dtype=torch.long, device=device)
        output, nce_loss = model(inputs, copy_label, missing_mode, epoch=epoch, opt=opt)

        loss_re = loss_fn(output, label)
        # NCE warmup scheduling
        nce_max = getattr(opt, 'nce_max_weight', 0.1)
        nce_warm = max(1, getattr(opt, 'nce_warmup_epochs', 5))
        nce_w = nce_max * min(1.0, float(epoch) / float(nce_warm))
        loss = loss_re + nce_w * nce_loss
        losses.update(loss.item(), batchsize)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        train_pbar.set_description('train')
        train_pbar.set_postfix({
            'epoch': '{}'.format(epoch),
            'loss': '{:.5f}'.format(losses.value_avg),
            'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
        })

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)

    return train_results


def evaluate(model, eval_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(eval_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            inputs = {
                'V': data['vision'].to(device),
                'A': data['audio'].to(device),
                'T': data['text'].to(device),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device).to(torch.bool),
                    'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device).to(torch.bool),
                    'T': []
                }
            }
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = inputs['V'].shape[0]

            missing_mode = data['missing_mode'].to(device) if 'missing_mode' in data else torch.full((inputs['V'].size(0),), 6, dtype=torch.long, device=device)
            output, _ = model(inputs, None, missing_mode)
            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            test_pbar.set_description('eval')
            test_pbar.set_postfix({
                'epoch': '{}'.format(epoch),
                'loss': '{:.5f}'.format(losses.value_avg),
                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            })

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        valid_results = metrics(pred, true)

    return valid_results


def test(model, test_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(test_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            inputs = {
                'V': data['vision'].to(device),
                'A': data['audio'].to(device),
                'T': data['text'].to(device),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device).to(torch.bool),
                    'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device).to(torch.bool),
                    'T': []
                }
            }
            ids = data['id']
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = inputs['V'].shape[0]

            missing_mode = data['missing_mode'].to(device) if 'missing_mode' in data else torch.full((inputs['V'].size(0),), 6, dtype=torch.long, device=device)
            output, _ = model(inputs, None, missing_mode)
            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({
                'epoch': '{}'.format(epoch),
                'loss': '{:.5f}'.format(losses.value_avg),
                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            })

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        if epoch == 11:
            calculate_u_test(pred, true)
        test_results = metrics(pred, true)

    return test_results


if __name__ == '__main__':
    main(opt)