root_path = '/yunity/tkerby2/Local_CorEx/Local_CorEx_Experiments/experiments/paper_neural_networks/mnist_classifier/model_output'
rep = 5

conf = dict(
    classifier = dict(
        lr = 5e-4,
        bs = 32,
        num_workers = 47,
        hidden_layers = [500, 400, 300],
        drop_out_p = 0.5,
        save_path = root_path + f'/do_classifier/rep_{rep}',
        patience = 10,
        use_batch_norm = True,
        no_act_1st_layer = False,
        logger = dict(
            # name = 'tb_logs',
            version = f'bs_128_lr_5em4_3_do_5_bn_T_{rep}'
        ),
        trainer = dict(
            devices = 4,
            max_epochs = 200,
            accelerator = 'gpu',
            strategy = 'ddp', #'ddp_find_unused_parameters_true',
            refresh_rate = 50
        ),
        start_with_init_weights = True,
        save_initial_weights = False,
        save_initial_weights_path = root_path + f'/replicates/clf_weights_hs_300_bn_T_mlp_{rep}.pth'
    ),
    autoencoder = dict(
        lr = 2e-3,
        bs = 32,
        num_workers = 47,
        encoder_layers = [500, 400, 300],
        decoder_layers = [300, 500],
        drop_out_p = .0,
        save_path = root_path + f'/do_classifier/rep_{rep}',
        patience = 10,
        use_batch_norm = True,
        logger = dict(
            # name = 'tb_logs',
            version = f'ae_bs_128_lr_2em3_hs_300_{rep}'
        ),
        trainer = dict(
            devices = 4,
            max_epochs = 200,
            accelerator = 'gpu',
            strategy = 'ddp',
            refresh_rate = 50
        ),
        load_classifier_path = root_path + f'/do_classifier/rep_{rep}/mnist_clf_epoch=096-val_loss=0.0029.ckpt'
    )
)