import os
import numpy as np
import torch
import random
import librosa as lb

from common.utils.pathUtils import createFullPathTree, loadPickle


class AudioDataset(torch.utils.data.Dataset):
    """
    Dataset for load wav files from multiple blob containers
    Takes a label from first character of the wav file name if first charcter is a number
    """

    def __init__(self, args, logger, mode, transform=None):
        """
        Loads from .txt files which are simply lists of file names
        Args:
            args - run arguments
            logger - run logger
            mode (string): Options includes `train`, `val`, or `test` mode.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for AudioDataset".format(mode)
        self.mode = mode
        self.test = True if mode == 'test' else False
        self.args = args
        self.logger = logger
        self.transform = transform

        bin_path_name = "data_{}_bin_path".format(mode.lower())
        assert hasattr(self.args, bin_path_name), "Config does not have the bin_data path data.{}".format(bin_path_name)
        bin_paths = getattr(self.args, bin_path_name)

        view_file_name = "data_{}_view_files".format(mode.lower())
        assert hasattr(self.args, view_file_name), "Config does not have the bin_data path data.{}".format(view_file_name)
        view_files = getattr(self.args, view_file_name)

        assert len(view_files) == len(bin_paths), "Expect len({}) == len({}) - got {} != {}".format(
          bin_path_name, view_file_name, len(view_files), len(bin_paths))

        self.clipList = []
        for bin_path, view_file in zip(bin_paths, view_files):
            self.construct_loader(bin_path, view_file)
        
        
    def construct_loader(self, binPath, viewFile):
        pathToViewFile = createFullPathTree(self.args.root_dir, viewFile)
        assert os.path.exists(pathToViewFile), "View file {} does not exist".format(
            pathToViewFile
        )
        files = []
        with open(pathToViewFile, 'r') as fp:
          for f in fp.readlines():
            label = int(f[0]) if f[0].isnumeric() else 0
            files.append((createFullPathTree(self.args.root_dir, binPath, f.strip()), label))

        self.clipList.extend(files)
        self.logger.info("Loading {} files from {} Total {}".format(len(files), viewFile, len(self.clipList)))

    def loadClip(self, index):
        clip = None
        label = None
        if index >= len(self.clipList):
            self.logger.debug("AudioLoader index is out of range Expect in range 0 - {}".format(len(self.clipList)))
            return clip, label
      
        fPath, label = self.clipList[index]
        if os.path.exists(fPath):
            clip, sr = lb.load(fPath, sr=self.args.data_samp_rate)
            if len(clip) != int(sr*self.args.data_pad_duration):
                if len(clip) < (sr*self.args.data_pad_duration)/2:
                    self.logger.warning('audio file {} is only {:0.2f} long'.format(fPath, len(clip)/sr))
                    x = np.zeros(sr*self.args.data_pad_duration, dtype='float32')
                    x[:len(clip)] = clip
                    clip = x
                if len(clip) > (sr*self.args.data_pad_duration):
                    self.logger.warning('audio file {} is too long: {:0.2f} seconds. only first {} seconds will be used'.format(fPath, len(clip)/sr, self.args.data_pad_duration))
                    clip = clip[:sr*self.args.data_pad_duration]
            clip = torch.tensor(clip)
            if sr != self.args.data_samp_rate:
                self.logger.debug("Got samp rate {} Expect {} while loading  {}".format(sr, self.args.data_samp_rate, fPath))
                clip = None
                label = None
        else:
            self.logger.debug("Did not find file {}".format(fPath))
      
        return clip, label
        
    def __getitem__(self, index):
        """
        Return the clip at index. If the clip cannot be loaded randomly finds another clip
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            clip - tensor of the clip
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        # If given index is invalid will pick another random index
        for _ in range(self.args.data_num_retry):
            clip, label = self.loadClip(index)
            if clip is None:
                index = random.randint(0, len(self.clipList) - 1)
                continue

            if self.transform is not None:
              clip = self.transform(clip)
            return clip, label, index
        else:
            raise RuntimeError(
                "Failed to fetch clip after {} retries.".format(
                    self.args.data_num_retry
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of clips in the dataset.
        """
        return len(self.clipList)
