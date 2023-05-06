def set_template(args):
    # Set the templates here

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
        args.epochs = 300
        args.loss = '1*MSE'
        args.lr = 1e-4
        args.final_lr = 1e-5
        args.lr_adjust = 'logarithmic'
        args.batch_size = 32
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