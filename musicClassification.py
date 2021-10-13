from pathlib import Path
import json
import subprocess
import time

import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score

# from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch

from barlowtwins.main import BarlowTwins
from barlowtwins.audioDataset import AudioDataset
from barlowtwins.audioTransformer import AudioTransformer

# from common.utils.pathUtils import createFullPathTree, ensureDir, savePickle, loadPickle
from common.utils.logger import CreateLogger

import logging
import azureml.core.authentication as authLog
import msrest.http_logger as http_logger
from msrest.universal_http.__init__ import _LOGGER as universalHttpLogger
from msrest.service_client import _LOGGER as serviceLogger
from urllib3.connectionpool import log as urllib3Logger


class MusicClassifier(object):
    def __init__(self, args):
        self.args = args
        self.logger = None

    def loggerWorkaround(self, azureLogger, name):
        '''
        Workaround around for azure loggers that by default spew debug logging that flood the output
        Simply set logging level to WARN
        '''
        before = azureLogger.getEffectiveLevel()
        azureLogger.setLevel(logging.WARNING)
        self.logger.info("{} logger workaround Loglevel Before {} After {}".format(
            name, before, azureLogger.getEffectiveLevel()))
    
    def loggerWorkaroundAll(self):

        # Workarounds for issue in S/C cluster that gets a wierd loglevel
        self.loggerWorkaround(authLog.module_logger, 'AzureAuthority')
        self.loggerWorkaround(http_logger._LOGGER, "http logger")
        self.loggerWorkaround(logging.getLogger("azureml"), "azureml logger")
        universalHttpLogger.debug("universalHttpLogger Debug Configuring requets Before")
        universalHttpLogger.info("universalHttpLogger INFO Configuring requets Before")
        self.loggerWorkaround(universalHttpLogger, "universal logger")
        universalHttpLogger.debug("universalHttpLogger DEBUG Configuring requets Before")
        self.loggerWorkaround(serviceLogger, "serviceLogger")
        self.loggerWorkaround(urllib3Logger, "urllib3 logger")


    def train(self, gpu=None):
        with CreateLogger(self.args, logger_type=self.args.logger_type) as logger:
            self.logger = logger
            self.loggerWorkaroundAll()
            self.args.checkpoint_dir = Path(self.args.output_dir)

            train(self.args, logger)

    def eval(self, gpu=None):
        with CreateLogger(self.args, logger_type=self.args.logger_type) as logger:
            self.logger = logger
            self.loggerWorkaroundAll()
            self.args.checkpoint_dir = Path(self.args.output_dir)

            eval_validation_set(self.args, logger)


def train(args, logger):
    model =  musicClassifier(args, logger)
    logger.info('loaded music classifier model')
    logger.debug(model)

    # train on gpu if available
    dev, model = get_device(logger, model)

    # automatically resume from checkpoint if it exists
    model = load_checkpoint(args, logger, model)

    # load datasets
    dataset_train = AudioDataset(args=args, logger=logger, mode='train', transform=AudioTransformer(args, logger))
    dataset_val = AudioDataset(args=args, logger=logger, mode='val', transform=AudioTransformer(args, logger))
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        sampler=sampler_train)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=args.batch_size, 
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
        )

    # prepare for training
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    early_stopper = EarlyStopper(args)

    start_time = time.time()
    logger.info('start training ..')
    for epoch in range(0, args.epochs):
        for step, ((x1, _), y1, _) in enumerate(loader_train, start=epoch * len(loader_train)):
            y1 = y1.type(torch.float).to(dev)
            x1 = x1.to(dev)

            optimizer.zero_grad()
            x1 = model.forward(x1)
            loss = criterion(x1,y1)
            loss.backward()
            optimizer.step()

            if step %args.plot_freq == 0:
                logger.log_row(name='loss', step=step, loss=loss.item())

            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step,
                            loss=loss.item(),
                            time=int(time.time() - start_time))
                logger.info(json.dumps(stats))
                if dev==torch.device("cuda"):
                    logger.debug(subprocess.check_output("nvidia-smi", shell=True, universal_newlines=True))

        # save checkpoint if best accuracy on validation set
        if (epoch % args.data_epoch_checkpoint_freq) == 0:

            results = evaluate(model, loader_val, dev)
            logger.info('epoch: {}, accuracy {:0.3f}, best accuracy {:0.3f}'.format(epoch+1, results['accuracy'], early_stopper.best_accuracy))
            logger.log_row(name='accuracy', epoch=epoch, accuracy=results['accuracy'])
            logger.log_row(name='recall', epoch=epoch, accuracy=results['recall'])
            logger.log_row(name='precision', epoch=epoch, accuracy=results['precision'])

            # save checkpoint
            if results['accuracy']>early_stopper.best_accuracy:
                statedict = model.module.state_dict() if (torch.cuda.device_count()>1) else model.state_dict()
                state = dict(epoch=epoch + 1, model=statedict,
                            optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / 'best_checkpoint.pth')
                logger.info('checkpoint saved')

            # stop early if validation accuracy does not improve
            stop_early = early_stopper.step(results['accuracy'], epoch+1)
            if stop_early:
                return


def eval_validation_set(args, logger):
    model =  musicClassifier(args, logger)
    logger.info('loaded music classifier model')
    logger.debug(model)

    # run on gpu if available
    dev, model = get_device(logger, model)

    # load checkpoint
    model = load_checkpoint(args, logger, model)

    # load datasets
    dataset_val = AudioDataset(args=args, logger=logger, mode='val', transform=AudioTransformer(args, logger))
    loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=args.batch_size, 
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
        )
    logger.info('start evaluating ..')
    results = evaluate(model, loader_val, dev)
    logger.info('accuracy {:0.3f}, recall {:0.3f}, precision {:0.3f}, '.format(results['accuracy'], results['recall'], results['precision'],))
    return results


def load_checkpoint(args, logger, model):
    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / args.checkpoint_name).is_file():
        ckpt = torch.load(args.checkpoint_dir / args.checkpoint_name,
                        map_location='cpu')
        [missing_keys, unexpected_keys ] = model.load_state_dict(ckpt['model'], strict=False)
        for missed_key in missing_keys:
            if not missed_key.startswith('backbone.fc'):
                raise ValueError('Found missing keys in checkpoint {}'.format(missing_keys))
        for unexpected_key in unexpected_keys:
            if not ((unexpected_key.startswith('bn.')) or (unexpected_key.startswith('projector.'))):
                raise ValueError('Found unexpected keys in checkpoint {}'.format(unexpected_keys))        
        logger.info('checkpoint loaded from {}'.format(args.checkpoint_dir / args.checkpoint_name))
    else:
        logger.info('no checkpoint loaded')
    return model


def get_device(logger, model):
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        model.to(dev)
        if torch.cuda.device_count()>1:
            model = torch.nn.parallel.DataParallel(model)
            logger.info('train on cude with data parallel on {} devices'.format(torch.cuda.device_count()))
        else:
            logger.info('train on cude without data parallel')
    else:
        dev = torch.device("cpu")
        logger.info('train on cpu')
    return dev, model


def evaluate(model, loader, dev):
        model.eval()
        with torch.no_grad():
            yy = [ [model(x1.to(dev)).cpu().numpy()>0.5, y1.cpu().numpy()] for ((x1, _), y1, _) in loader]
        yy = np.concatenate( yy, axis=1 )
        y_pred = yy[0,:].reshape(-1,1)
        y_true = yy[1,:].reshape(-1,1)
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        results = {'accuracy': accuracy, 'recall': recall, 'precision': precision}
        return results


class EarlyStopper(object):          
    def __init__(self, args):
        self.patience = args.early_stop_patience
        self.args = args
        self.best_accuracy = -1e10
        self.best_epoch = 0
        self.cnt = -1
        
    def step(self, accuracy, epoch):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = epoch
            self.cnt = -1          
        self.cnt += 1 

        if self.cnt >= self.patience:
            stop_early = True
            return stop_early
        else:
            stop_early = False
            return stop_early


class musicClassifier(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        self.args = args
        
        barlow_model = BarlowTwins(self.args, logger)
        self.backbone = barlow_model.backbone

        if barlow_model.lastLayerName == 'fc1':
            self.backbone.fc1 = nn.Linear(barlow_model.lastLayerSize, 1, bias=True)
        elif barlow_model.lastLayerName == 'fc':
            self.backbone.fc = nn.Linear(barlow_model.lastLayerSize, 1, bias=True)
        else:
            raise ValueError('Last layer name {} unkown'.format(barlow_model.lastLayerName))

    def forward(self, x):
        x = self.backbone(x)
        x = torch.sigmoid(x).view(-1)
        return x

