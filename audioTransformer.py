from torch import nn
from torchaudio import transforms as torchAudioTransforms 
import barlowtwins.audioTransforms as localTransforms

class AudioTransformer(object):
  '''
  Creates transformers for the BarlowTwins network.
  An input audio tensor is transformed twice for each pathway in the network
  The transforms are specified in the .yaml file using data_transforms_1 and data_transforms_2
  Each is an array of tuples:
    [name, arguments]
  name is the transform name. Can be either a totch audio transform or one found in audioTransforms.py
  arguments is a dict of argName: value  that will be accepted by the transform
  '''
  def __init__(self, args, logger):
    self.args = args
    self.logger = logger

    self.transform_1 = self.createTransforms(self.args.data_transforms_1)
    self.transform_2 = self.createTransforms(self.args.data_transforms_2)

  def createTransforms(self, transformSpec):
    '''
    Creates a transform from a transform Spec
    transformSpec (list) transformName, dict of transform arguments
    '''
    transforms = []
    for t, kwargs in transformSpec:
      found = False
      for tCollection in [torchAudioTransforms, localTransforms]:
        if hasattr(tCollection, t):
          trans = getattr(tCollection, t)
          transforms.append(trans(**kwargs))
          found = True
          break

      if not found:
        self.logger.info("{} is not an Audio Transform - skipping".format(t))
    
    return nn.Sequential(*transforms)

  
    def __call__(self, x):
        y1 = self.transform_1(x)
        y2 = self.transform_2(x)
        return y1, y2


