config = {
    'sgd_lr': 0.001,
    'sgd_epochs': 10,
    # 'sgd_epochs': 2,
    'adam_lr': 0.0001,
    'adam_epochs': 10,
    # 'adam_epochs': 2,
    'lr_scaledown': 0.1,
    'epoch_scale_up_for_lr_scale_down': 3,
    'warmup_proportion': 0.02,
    'steps_per_epoch': 100,
    'default_bits': 2,
    'weight_decay': 0.0,
    'other_bits': [],
    'bit_to_k_map': {
        4: 18,
        2: 6.5,
        1: 2,
    },
    'lr_jitter_scale': 0.01,
    'cache_data': True,
    'get_change_point_stats': True,
    'path': 'results',
    'get_change_point_stats': False,
    'epochs_interval': 1,
}