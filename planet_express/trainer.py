import os
from torch.utils.data import DataLoader
from planet_express.utils import setup_logger, get_ddp_state, get_dataloader_sampler


class Trainer:
    """
    This is the main PlanetExpress trainer class. It is responsible for training and evaluating
    the model. It is also responsible for saving the model and the optimizer state.

    The way to use this is to extend it and override the following methods:
        - Initializing the model (init_model)
        - Initializing the train and validation datasets (init_datasets)
        - Define the loss function (compute_loss)
    """

    def __init__(self, args):
        self.args = args
        
        self.model = None
        self.train_dataset, self.train_dataloader = None, None
        self.val_dataset, self.val_dataset = None, None
        
        self.init_model()
        self.init_datasets()
        
        self.logger_path = os.path.join(self.args.output_dir, "train.log")
        self.logger = setup_logger(self.logger_path)
        self.ddp_state = get_ddp_state()
        if self.ddp_state.is_master_process:
            if os.path.exists(self.logger_path):
                os.remove(self.logger_path)
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)
        self.device = f"cuda:{self.ddp_state.local_rank}"
        if self.train_dataset is not None:
            sampler = get_dataloader_sampler(self.ddp_state, self.train_dataset)
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=args.batch_size,
                shuffle=True if sampler is None else False,
                sampler=sampler,
                num_workers=args.num_workers,
                collate_fn=self.collate_function,
            )
    
    def init_model(self):
        raise NotImplementedError

    def init_datasets(self):
        raise NotImplementedError

    def collate_function(self, batch):
        raise NotImplementedError