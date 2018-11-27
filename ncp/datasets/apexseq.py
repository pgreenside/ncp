
import numpy as np

import sys
sys.path.insert(0, '/next/u/pgreens/git/ncp')
from ncp import tools

GIT_REPO='/next/u/pgreens/git/squirl/squirl'
sys.path.insert(0, GIT_REPO)
import featurize
import data_loader
import splitters

def generate_apexseq_dataset(
      normalize=True, 
      datapath='/next/u/pgreens/projects/dna_design/data/apexseq_data/'):

  apexseq_dataset = data_loader.ApexSeqLoader()
  apexseq_dataset.load({'data_path': datapath}) 

  featurizer = getattr(featurize, 'MultiFeaturizer')()
  featurizer_args =  {
      "featurizers": "OneHotFlatFeaturizer,DNAStatsFeaturizer,MisMatchFeaturizer",
      "mismatch_file": "/next/u/pgreens/projects/dna_design/data/apexseq_data/mismatches/apex_seqs_11_12_18_0_to_3_mismatches_with_locations.txt"
  }
  X, X_labels = featurizer.featurize(apexseq_dataset.input_sequences,
                                     featurizer_args)
  Y = np.array(apexseq_dataset.input_labels)

  # Z score Y values
  Y = (Y - np.mean(Y)) / np.std(Y)

  splitter = splitters.RandomSplitter()
  (train_idx, test_idx, valid_idx) = splitter.split(apexseq_dataset, {})

  # Currently 2d and 3d inputs are returned as a list
  X_train = X[0][train_idx,:]
  X_test = X[0][test_idx,:]
  X_valid = X[0][valid_idx,:]
  Y_train = Y[train_idx, None]
  Y_test = Y[test_idx, None]
  Y_valid = Y[valid_idx, None]

  print('train: %s, valid %s, test %s'%(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

  domain = np.linspace(np.vstack([Y_train, Y_test]).min(),
                       np.vstack([Y_train, Y_test]).max(),
                       1000)
  train = tools.AttrDict(inputs=X_train, targets=Y_train)
  test = tools.AttrDict(inputs=X_test, targets=Y_test)
  return tools.AttrDict(domain=domain, train=train, test=test, target_scale=1)



