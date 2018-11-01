
import numpy as np

from ncp import tools

def generate_osseq_dataset(
      normalize=True, 
      datapath='/next/u/pgreens/projects/dna_design/data/osseq_data/numpy_datasets'):
  X_train = np.load('%s/osseq_onehotflat_mismatch_dnastats_X_train_5797.npy'%datapath)
  X_test = np.load('%s/osseq_onehotflat_mismatch_dnastats_X_test_1450.npy'%datapath)
  if normalize:
    Y_train = np.load('%s/osseq_onehotflat_mismatch_dnastats_Y_std_norm_train_5797.npy'%datapath)
    Y_test = np.load('%s/osseq_onehotflat_mismatch_dnastats_Y_std_norm_test_1450.npy'%datapath)
  else:
    Y_train = np.load('%s/osseq_onehotflat_mismatch_dnastats_Y_train_5797.npy'%datapath)
    Y_test = np.load('%s/osseq_onehotflat_mismatch_dnastats_Y_test_1450.npy'%datapath)
  domain = np.linspace(np.vstack([Y_train, Y_test]).min(),
                       np.vstack([Y_train, Y_test]).max(),
                       1000)
  train = tools.AttrDict(inputs=X_train, targets=Y_train)
  test = tools.AttrDict(inputs=X_test, targets=Y_test)
  return tools.AttrDict(domain=domain, train=train, test=test, target_scale=1)
