import argparse
import copy
import datetime


from barlowtwins.launch import launch_job
from barlowtwins.main import Trainer
from common.utils.yamlConfig import YamlConfig
from common.utils.logger import CreateLogger


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
        logger.info(config.ReportConfig())
        logger.log_value('title', args.log_title, 'Run Title entered when job started')

        
        trainer = Trainer(args)
        launch_job(args=args, init_method=None, func=trainer.run)        

if __name__ == "__main__":
  main()
