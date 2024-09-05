
def str2arg(inp):
    return sorted([tuple(map(int, i.split(','))) for i in inp.split(';')], key=lambda x: x[0]) 

def str2dict_conv(inp):
    lst = inp.split(',')
    dic = {}
    dic['size'] = int(lst[0])
    dic['out'] = int(lst[1])
    dic['pad'] = int(lst[2])
    dic['stride'] = int(lst[3])
    dic['dil'] = int(lst[4])
    return dic

def set_up_args(args, filename):
    args = model_setup(args, filename)
    args = adversarial_setup(args, filename)
    args = adv_training_setup(args)
    args = adv_pretrained_setup(args)
    args = dataclass_setup(args, filename)
    args = channel_setup(args, filename)
    args = save_setup(args, filename)
    check_for_exceptions(args, filename)
        
    return args

def save_setup(args, filename):
    if filename not in ['compute_sdn_delta.py', 'run_avg_l2norm.py']:
        if filename == 'run_bicoatnet.py':
            save_opt = ArgOrganizer(n_channels_f=args.n_channels_f,
                                    n_channels_s=args.n_channels_s,
                                    input_size=args.input_size,
                                    num_blocks=args.num_blocks,
                                    num_channels=args.num_channels)
        elif filename == 'run_basecnn.py' or filename == 'run_pretraining.py':
            save_opt = ArgOrganizer(n_channels=args.n_channels,
                                    input_size=args.input_size,
                                    conv1=args.conv1,
                                    pool1=args.pool1,
                                    conv2=args.conv2,
                                    pool2=args.pool2,
                                    fc2=args.fc2)
        elif filename == 'run_pretrained.py':
            save_opt = ArgOrganizer(model_name=args.model_name,
                                    n_channels=args.n_channels)
        elif filename == 'run_biattncnn.py':
            save_opt = ArgOrganizer(n_channels_f=args.n_channels_f,
                                    n_channels_s=args.n_channels_s,
                                    model_dim=args.model_dim,
                                    dim_k=args.dim_k,
                                    dim_q=args.dim_q,
                                    n_heads=args.n_heads,
                                    conv1=args.conv1,
                                    conv2=args.conv2,
                                    conv3=args.conv3,
                                    conv4=args.conv4,
                                    pool1=args.pool1,
                                    pool2=args.pool2,
                                    fc=args.fc,
                                    input_size=args.input_size,
                                    patch_size=args.patch_size)
        elif filename == 'run_attncnn.py':
            save_opt = ArgOrganizer(n_channels=args.n_channels,
                                    model_dim=args.model_dim,
                                    dim_k=args.dim_k,
                                    dim_q=args.dim_q,
                                    n_heads=args.n_heads,
                                    conv1=args.conv1,
                                    conv2=args.conv2,
                                    conv3=args.conv3,
                                    conv4=args.conv4,
                                    pool1=args.pool1,
                                    pool2=args.pool2,
                                    fc=args.fc,
                                    input_size=args.input_size,
                                    patch_size=args.patch_size)
        elif filename == 'run_coatnet.py':
            save_opt = ArgOrganizer(n_channels=args.n_channels,
                                    input_size=args.input_size,
                                    num_blocks=args.num_blocks,
                                    num_channels=args.num_channels)
            
        elif filename == 'run_bihpf.py':
            raise NotImplementedError('BiHPF CURRENTLY NOT IMPLEMENTED')
        elif filename == 'run_synmixer.py':
            raise NotImplementedError('SynMixer CURRENTLY NOT IMPLEMENTED')
        elif filename == 'run_lrf.py':
            raise NotImplementedError('LRF CURRENTLY NOT IMPLEMENTED')
        else:
            raise NotImplementedError('SAVE OPT FOR OTHER MODELS MUST BE IMPLEMENTED FIRST. go to argutils to det save args')
        
        save_opt.model_out = args.model_out
        args.save_opt = save_opt
    return args
    
    
def check_for_exceptions(args, filename):
    pass

def channel_setup(args, filename):
    
    greyscale_opt = ArgOrganizer(greyscale_processing=args.greyscale_processing,
                                 greyscale_fourier=args.greyscale_fourier)
    
    args.greyscale_opt = greyscale_opt
    
    if args.greyscale_opt.greyscale_processing:
        args.greyscale_opt.greyscale_fourier = False
    if filename != 'run_avg_l2norm.py':
        if args.greyscale_opt.greyscale_processing:
            
            if args.transform == 'band_cooccurrence':
                raise ValueError('GREYSCALE PROCESSING NOT AVAILABLE WITH INTER-BAND COOCCURENCE METHODS. Change transform or processing type')
            else:
                args.n_channels = 1
        else:
            if filename in ['run_bicoatnet.py', 'run_biattncnn.py']:
                args.n_channels_s = 3
                args.n_channels_f = 1 if args.greyscale_opt.greyscale_fourier else 3
            elif args.transform in ['basic_fr_attn_cnn', 'real_nd_fourier', 'augmented_nd_fourier']:
                args.n_channels = 1 if args.greyscale_opt.greyscale_fourier else 3
            elif args.transform == 'band_cooccurrence':
                args.n_channels = 12
                args.input_size = 256
            else:
                args.n_channels = 3
    
    return args

def model_setup(args, filename):
    if filename == 'run_basecnn.py':
        if args.transform == 'band_cooccurence':
            args.model_name = 'CooccurrenceCNN'
            args.model_out = 'cooccurrence_cnn'
        else:
            args.model_name = 'BaseCNN'
            args.model_out = 'base_cnn'
    elif filename == 'run_pretrained.py':
        if args.transform == 'augmented_pretrained_imgnet':
            args.model_dir_name = 'AugImgNetCNN'
            args.model_out = 'pretrained'
        else:
            args.model_dir_name = 'ImgNetCNN'
            args.model_out = 'pretrained'
    return args

def adversarial_setup(args, filename):

    if args.run_cw_test == True and args.run_auc_test == True:
        raise ValueError('Only one of [cw test] and [auc test] can be performed at a time.')

    adversarial_opt = ArgOrganizer(adversarial=args.adversarial,
                                spectral_delta_path=args.spectral_delta_path,
                                power_dict_path=args.power_dict_path,
                                greyscale_processing=args.greyscale_processing,
                                spatial_adv_type=args.spatial_adv_type,
                                attack_compression=args.attack_compression,
                                compression_rate=args.attack_compression_rate,
                                consecutive_attack_compr=args.consecutive_attack_compr,
                                is_targeted=args.is_targeted,
                                scale_cad=args.scale_cad_for_asp)
    
    if adversarial_opt.adversarial:

        spatial_attack_params = ArgOrganizer(eps=args.eps)
        surrogate_model_params = ArgOrganizer()
        surrogate_model_params.surrogate_model = args.surrogate_model
        surrogate_model_params.surrogate_input_size = args.surrogate_input_size
        surrogate_model_params.surrogate_transform = args.surrogate_transform
        adversarial_opt.surrogate_model_params = surrogate_model_params
        if adversarial_opt.spatial_adv_type not in ['fgsm',
                                                    'cw',
                                                    'rcw',
                                                    'far']:
            spatial_attack_params.alpha = args.alpha
        if adversarial_opt.spatial_adv_type == 'far':
            spatial_attack_params.eta = args.eta
        if adversarial_opt.spatial_adv_type in ['jifgsm']:
            spatial_attack_params.N = args.N
            spatial_attack_params.steps = args.steps
        if adversarial_opt.spatial_adv_type == 'jifgsm':
            spatial_attack_params.jifgsm_compr_type = args.jifgsm_compr_type
        if adversarial_opt.spatial_adv_type in ['cw',
                                                'rcw']:
            spatial_attack_params.c = args.c
            spatial_attack_params.kappa = args.kappa
            spatial_attack_params.steps = args.steps
            spatial_attack_params.attack_lr = args.attack_lr
            spatial_attack_params.n_starts = args.n_starts
            spatial_attack_params.verbose_cw = args.verbose_cw
            spatial_attack_params.target_mode = args.target_mode
            if adversarial_opt.spatial_adv_type in ['rcw']:
                spatial_attack_params.rcw_comp_lower_bound = args.rcw_comp_lower_bound
                spatial_attack_params.rcw_beta = args.rcw_beta
        if adversarial_opt.spatial_adv_type in ['jifgsm', 'cvfgsm', 'far']:
            if args.is_targeted:
                spatial_attack_params.target_mode = args.target_mode
        if adversarial_opt.spatial_adv_type == 'far':
                spatial_attack_params.far_jpeg_quality = args.far_jpeg_quality
        adversarial_opt.spatial_attack_params = spatial_attack_params
    if filename == 'run_pretrained.py':
        input_size = 299 if args.model_name == 'xception' else 224
        adversarial_opt.input_size = input_size   
    else:
        adversarial_opt.input_size = args.input_size
    
    args.adversarial_opt = adversarial_opt
    return args

def adv_training_setup(args):
    
    adversarial_training_opt = ArgOrganizer(adversarial_training=args.adversarial_training,
                                            adv_training_type=args.adv_training_type,
                                            attacks_for_training=args.attacks_for_training,
                                            training_eps=args.training_eps)

    args.adversarial_training_opt = adversarial_training_opt

    return args

def adv_pretrained_setup(args):
    
    adversarial_pretrained_opt = ArgOrganizer(adversarial_pretrained=args.adversarial_pretrained,
                                              adv_pretrained_protocol=args.adv_pretrained_protocol)

    args.adversarial_pretrained_opt = adversarial_pretrained_opt

    return args
    
    
def dataclass_setup(args, filename):

    # sdn delta computation does not use an gradient-based optimization algorithm

    if filename != 'run_avg_l2norm.py':

        if args.optim in ['RADAM', 'radam']: # uses RAdam: https://arxiv.org/pdf/1908.03265.pdf
            args.optim = {
                'optim': 'radam',
                'warm_up' : True,
                'weight_decay': 1e-5,
                'lr' : args.lr,
                'scheduler': 'reduce_on_plateau',
            }

        elif args.optim in ['sgdn', 'SGDN']:
            args.optim = {
                'optim': 'sgdn',
                'warm_up' : False,
                'weight_decay': 1e-5,
                'scheduler': 'exp_lr',
                'momentum': 0.9,
                'nesterov': True, # uses nesterov momentum: http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
                'lr' : args.lr
            }
        
        elif args.optim in ['sgd', 'SGD']:
            args.optim = {
                'optim': 'sgd',
                'warm_up' : False,
                'weight_decay': 1e-5,
                'scheduler': 'reduce_on_plateau',
                'momentum': 0.0,
                'nesterov': False,
                'lr' : args.lr
            }
            
        elif args.optim in ['adam', 'ADAM']:
            args.optim = {
                'optim': 'adam',
                'warm_up' : False,
                'scheduler': 'reduce_on_plateau',
                'weight_decay': 1e-5,
                'lr' : args.lr
            } 

    return args
    

class ArgOrganizer:
    
    def __init__(self, *args, **kwargs):
        for name, value in  kwargs.items():
            setattr(self, name, value)

if __name__ == '__main__':
    pass