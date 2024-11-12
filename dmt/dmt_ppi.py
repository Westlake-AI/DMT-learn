import os

class DMT_PPI():
    def __init__(self, description = "KeAP20_string",
        ppi_path = './data/protein.actions.SHS27k.STRING.txt',
        pseq_path = "data/protein.SHS27k.sequences.dictionary.tsv",
        vec_path = "data/PPI_embeddings/protein_embedding_KeAP20_shs27k.npy",
        split_new = "True",
        split_mode = "dfs",
        train_valid_index_path = "./data/new_train_valid_index_json/STRING.dfs.fold1.json",
        use_lr_scheduler = "True",
        save_path = "./output/ppi",
        graph_only_train = "False",
        batch_size = 2048,
        epochs = 300,
        use_dmt = True,
        v_input = 100,
        v_latent = 0.01,
        sigmaP = 1.0,
        augNearRate = 100000,
        balance = 0.01,
        seed = 0) -> None:

        self.description = description

        self.ppi_path = ppi_path
        self.pseq_path = pseq_path
        self.vec_path = vec_path

        self.split_new = split_new
        self.split_mode = split_mode
        self.train_valid_index_path = train_valid_index_path
        self.use_lr_scheduler = use_lr_scheduler
        self.save_path = save_path
        self.graph_only_train = graph_only_train

        self.batch_size = batch_size
        self.epochs = epochs

        self.use_dmt = use_dmt
        self.v_input = v_input
        self.v_latent = v_latent
        self.sigmaP = sigmaP
        self.augNearRate = augNearRate
        self.balance = balance
        self.seed = seed

    def fit(self):
        os.system("python -u gnn_train_dmt.py \
                --description={} \
                --ppi_path={} \
                --pseq_path={} \
                --vec_path={} \
                --split_new={} \
                --split_mode={} \
                --train_valid_index_path={} \
                --use_lr_scheduler={} \
                --save_path={} \
                --graph_only_train={} \
                --batch_size={} \
                --epochs={} \
                --use_dmt={} \
                --v_input={} \
                --v_latent={} \
                --sigmaP={} \
                --augNearRate={} \
                --balance={} \
                --seed={}".format(self.description, self.ppi_path, self.pseq_path, self.vec_path, 
                        self.split_new, self.split_mode, self.train_valid_index_path,
                        self.use_lr_scheduler, self.save_path, self.graph_only_train, 
                        self.batch_size, self.epochs, self.use_dmt, self.v_input, self.v_latent, self.sigmaP, self.augNearRate, self.balance, self.seed))
        
    def transform(self):
        description = "test"

        ppi_path = self.ppi_path
        pseq_path = self.pseq_path
        vec_path = self.vec_path

        index_path = self.train_valid_index_path
        # path to checkpoint
        dir = "output/ppi/gnn_KeAP20_string_dfs_0_True_0.01_2024-11-09_15:54:54"
        gnn_model= dir + "/gnn_model_valid_best.ckpt"

        test_all = "True"

        os.system("python gnn_test_dmt.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --index_path={} \
            --gnn_model={} \
            --test_all={} \
            --use_dmt={} \
            --v_input={} \
            --v_latent={} \
            --sigmaP={} \
            --augNearRate={} \
            --balance={} \
            ".format(description, ppi_path, pseq_path, vec_path, 
                    index_path, gnn_model, test_all, self.use_dmt, self.v_input, self.v_latent, self.sigmaP, self.augNearRate, self.balance))
        
    def fit_transform(self):
        self.fit()
        self.transform()
