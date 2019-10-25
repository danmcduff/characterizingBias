
#----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

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
#tf_config['gpu_options.allow_growth']          = False     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
#env.CUDA_VISIBLE_DEVICES                       = '0'       # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL                        = '1'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

#----------------------------------------------------------------------------
# Official training configs, targeted mainly for CelebA-HQ.
# To run, comment/uncomment the lines as appropriate and launch train.py.

desc        = 'pgan'                                        # Description string included in result subdir name.
random_seed = 1000                                          # Global random seed.
dataset     = EasyDict()                                    # Options for dataset.load_dataset().
train       = EasyDict(func='train.train_progressive_gan')  # Options for main training func.
G           = EasyDict(func='networks.G_paper')             # Options for generator network.
D           = EasyDict(func='networks.D_paper')             # Options for discriminator network.
G_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for generator optimizer.
D_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for discriminator optimizer.
G_loss      = EasyDict(func='loss.G_wgan_acgan')            # Options for generator loss.
D_loss      = EasyDict(func='loss.D_wgangp_acgan')          # Options for discriminator loss.
sched       = EasyDict()                                    # Options for train.TrainingSchedule.
grid        = EasyDict(size='1080p', layout='random')       # Options for train.setup_snapshot_image_grid().

# Dataset (choose one).
desc += '-face';            dataset = EasyDict(tfrecord_dir='face'); train.mirror_augment = True

# Conditioning & snapshot options.
desc += '-cond'; dataset.max_label_size = 'full' # conditioned on full label

# Config presets (choose one).
desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 16; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}; sched.G_lrate_dict = {256: 0.0015, 512: 0.002, 1024: 0.003}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 12000

# Numerical precision (choose one).
desc += '-fp32'; sched.max_minibatch_per_gpu = {256: 16, 512: 8, 1024: 4}

#----------------------------------------------------------------------------
# Utility scripts.
# To run, uncomment the appropriate line and launch train.py.

#train = EasyDict(func='util_scripts.generate_fake_images', run_id=14, num_pngs=100); num_gpus = 1; desc = 'fake-images-' + str(train.run_id)
train = EasyDict(func='util_scripts.generate_fake_images', run_id=14, grid_size=[15,8], num_pngs=1, image_shrink=1); num_gpus = 1; desc = 'fake-grids-' + str(train.run_id)

#----------------------------------------------------------------------------
