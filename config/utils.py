import os
os.environ['MPLCONFIGDIR'] = './'


def print_options(opt, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        # message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '{:>25}: {}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)


def sync_options(opt, config):
    opt_dict = vars(opt)
    all_keys = set(list(opt_dict.keys()) + list(config.keys()))
    for n in all_keys:
        # config -> opt
        if getattr(opt, n, None) is None:
            if n in config.keys():
                setattr(opt, n, config[n])
        # opt -> config
        if n not in config.keys():
            config[n] = opt_dict[n]
        # opt first
        if getattr(opt, n) != config[n]:
            config[n] = opt_dict[n]
    return opt, config
