import argparse
import copy
import datetime
import socket
import os
import warnings
import torch
import torchvision
import torchaudio

from barlowtwins.launch import launch_job
from barlowtwins.main import Trainer
from common.utils.yamlConfig import YamlConfig
from common.utils.logger import CreateLogger
from common.utils.pathUtils import ensureDir

'''
entry point for hackerthon project using barlowTwin unsupervided learning
'''

def ParseArgs():
  parser = argparse.ArgumentParser("hack")

  parser.add_argument("--root_dir", type=str, help="Experiment root directory")
  parser.add_argument("--config_root", type=str,
                      help="Configuration directory")
  parser.add_argument("--config_files", nargs='+', help="Configuration File")
  parser.add_argument("--output_dir", type=str, help="Output directory")
  parser.add_argument("--input_file_storage", type=str, help="File storage director")
  parser.add_argument("--log_title", type=str, help="Run title", default='None')

  args = parser.parse_args()
  return args


def main():
  argsOrig = ParseArgs()
  args = ParseArgs()
  now = datetime.datetime.today()
  for config_file in argsOrig.config_files:
    args = copy.deepcopy(argsOrig)
    args.config_file = config_file
    with YamlConfig(args, now) as config:
      args = config.ApplyConfigFile(args)

      with CreateLogger(args, logger_type=args.logger_type) as logger:
        ensureDir(args.output_dir)
        logger.info(config.ReportConfig())
        logger.log_value('title', args.log_title, 'Run Title entered when job started')
        host_name = socket.gethostname() 
        host_ip = socket.gethostbyname(host_name)        


        logger.info("Starting on host {} host_ip {}".format(host_name, host_ip))
        logger.info("torch version {}".format(torch.__version__))
        logger.info("Torchvision version {}".format(torchvision.__version__))
        logger.info("TorchAudio version {}".format(torchaudio.__version__))
        logger.info("Cuda enabled {} num GPU {}".format(torch.cuda.is_available(), torch.cuda.device_count()))
        logger.info("Ignoring warnings")
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

        logger.info(config.ReportConfig())
        args.master_addr = host_ip
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = str(args.master_port)
        os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
        logger.info("MASTER_ADDR {} MASTER_PORT {} WORLD_SIZE {}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"], os.environ["WORLD_SIZE"]))
        
        trainer = Trainer(args)
        launch_job(args=args, init_method=None, func=trainer.run)        

if __name__ == "__main__":
  main()
