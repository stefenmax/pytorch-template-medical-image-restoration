def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('DDBPN') >= 0:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'

        args.data_test = 'Set5'

        args.batch_size = 20
        args.epochs = 1000
        args.decay = '500'
        args.gamma = 0.1
        args.weight_decay = 1e-4

        args.loss = '1*MSE'

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.decay = '150'

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = True

    if args.template.find('VDSR') >= 0:
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 41
        args.lr = 1e-1

    if args.template.find('AAPM') >= 0:
        args.data_train = 'AAPM'
        args.data_test = 'AAPM'
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        
    if args.template.find('RID') >= 0:
        args.data_train = 'AAPM'
        args.data_test = 'AAPM'
        args.model = 'RIDNET'
        args.n_feats = 64
        
    if args.template.find('FBPCONV') >= 0:
        args.data_train = 'AAPM'
        args.data_test = 'AAPM'
        args.model = 'FBPCONV'
        args.epochs = 100
        args.loss = '1*MSE'
        args.lr = 1e-2
        args.final_lr = 1e-3
        args.lr_adjust = 'logarithmic'
        args.batch_size = 1
        args.optimizer = 'SGD'
        args.momentum = 0.99
        
    if args.template.find('RED') >= 0:
        args.data_train = 'AAPM'
        args.data_test = 'AAPM'
        args.model = 'REDCNN'
        args.n_feats = 96
        args.patch_size = 55
        args.lr = 1e-4
        args.final_lr = 1e-5
        args.lr_adjust = 'logarithmic'
        args.loss = '1*MSE'
        
    if args.template.find('DDNET') >= 0:
        args.data_train = 'AAPM'
        args.data_test = 'AAPM'
        args.model = 'DDNET'
        args.patch_size = 64
        args.lr = 1e-4
        args.final_lr = 1e-5
        args.lr_adjust = 'logarithmic'
        args.loss = '1*MSE'
        args.batch_size = 5
