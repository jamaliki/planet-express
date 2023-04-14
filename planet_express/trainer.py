import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from planet_express.utils import setup_logger, get_ddp_state, get_dataloader_sampler, wrap_model, get_optimizer_class, \
    get_dtype, get_lr_scheduler, unwrap_state_dict
from planet_express.summary_writer_wrapper import SummaryWriterMPWrapper, SummaryData


class Trainer:
    """
    This is the main PlanetExpress trainer class. It is responsible for training and evaluating
    the model. It is also responsible for saving the model and the optimizer state.

    The way to use this is to extend it and override the following methods:
        - Initializing the model (init_model)
        - Initializing the train and validation datasets (init_datasets)
        - Define the train step (train_step)
    """

    def __init__(self, args):
        self.args = args

        # Force start method to be spawn
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except:
            pass

        self.model = None
        self.train_dataset, self.train_dataloader = None, None
        self.val_dataset, self.val_dataloader = None, None
        self.start_epoch = 0
        self.iter = 0

        self.dtype = get_dtype(self.args.dtype)
        
        self.init_model()
        self.init_datasets()
        
        self.logger_path = os.path.join(self.args.output_dir, "train.log")
        self.logger = setup_logger(self.logger_path)
        self.ddp_state = get_ddp_state()
        self.summary_writer = None
        if self.ddp_state.is_master_process:
            if os.path.exists(self.logger_path):
                os.remove(self.logger_path)
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)
            self.summary_writer = SummaryWriterMPWrapper(os.path.join(args.output_path, "summary"))
        self.device = f"cuda:{self.ddp_state.local_rank if self.ddp_state.is_ddp else args.device}"
        self.model = wrap_model(self.model, self.ddp_state, self.device)
        self.optimizer = get_optimizer_class(args.optimizer)(
            self.model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate
        )
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
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
        if self.val_dataset is not None:
            if self.ddp_state.is_master_process:
                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=2,
                    collate_fn=self.collate_function,
                )
    
    def init_model(self):
        raise NotImplementedError

    def init_datasets(self):
        raise NotImplementedError
    
    def train_step(self, batch):
        raise NotImplementedError

    def collate_function(self, batch):
        return torch.utils.data.default_collate(batch)
    
    def train(self):
        for epoch in range(self.start_epoch, self.args.num_epochs):
            if self.ddp_state.is_ddp:
                self.train_dataloader.sampler.set_epoch(epoch)
            self.model.train()
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    loss_dict = self.train_step(batch)
                loss = loss_dict["final_loss"]
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.iter += 1

                if self.ddp_state.is_master_process:
                    summary_data = [SummaryData("scalar", f"Loss/{k}", v.item()) for k, v in loss_dict.items()]
                    self.summary_writer.add(
                        self.iter,
                        *summary_data
                    )
                    if batch_idx % self.args.log_interval == 0:
                        self.logger.info(
                            f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item()}"
                        )
                        self.logger.info(f"Loss dict: {loss_dict}")

            self.model.eval()
            if self.val_dataloader is not None:
                total_loss, num_iters = 0, 0
                with torch.no_grad():
                    for batch in self.val_dataloader:
                        val_loss_dict = self.train_step(batch)
                        total_loss += val_loss_dict["final_loss"].item()
                        num_iters += 1
                self.logger.info(f"Epoch: {epoch} | Validation loss: {total_loss / num_iters}")
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        if self.ddp_state.is_master_process:
            model_path = os.path.join(self.args.output_dir, f"checkpoint_{epoch}.pt")
            save_dict = {
                "epoch": epoch,
                "iter": self.iter,
                "model": unwrap_state_dict(self.model.state_dict()),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "train_args": self.args,
            }
            torch.save(save_dict, model_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.start_epoch = checkpoint["epoch"]
        self.iter = checkpoint["iter"]
        self.args = checkpoint["train_args"]
