
#----------------------------------------------------------------------------

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

#----------------------------------------------------------------------------
# Paths.

data_dir = 'datasets'
result_dir = 'results'

#----------------------------------------------------------------------------
# TensorFlow options.

tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.

tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
env.TF_CPP_MIN_LOG_LEVEL                        = '1'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

#----------------------------------------------------------------------------
# Official training configs, targeted mainly for CelebA-HQ.
# To run, comment/uncomment the lines as appropriate and launch train.py.

desc        = 'pgan'                                        # Description string included in result subdir name.
random_seed = 1000                                          # Global random seed.
dataset     = EasyDict()                                    # Options for dataset.load_dataset().
G           = EasyDict(func='networks.G_paper')             # Options for generator network.
D           = EasyDict(func='networks.D_paper')             # Options for discriminator network.


#----------------------------------------------------------------------------

train = EasyDict(func='util_scripts.generate_fake_images', run_id=14, num_pngs=100); num_gpus = 1; desc = 'fake-images-' + str(train.run_id)

#----------------------------------------------------------------------------
