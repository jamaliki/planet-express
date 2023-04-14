import multiprocessing as mp
import os
from collections import namedtuple

from torch.utils.tensorboard import SummaryWriter


SummaryData = namedtuple("SummaryData", ["type", "name", "value"])


def run_summary_writer_server(queue, *args, **kwargs):
    summary_writer = SummaryWriterWrapper(*args, **kwargs)
    while True:
        data = queue.get()
        summary_writer.add(data["step"], *data["args"])


class SummaryWriterMPWrapper:
    def __init__(self, *args, **kwargs) -> None:
        self.queue = mp.Queue()
        self.server_process = mp.Process(
            target=run_summary_writer_server,
            args=(
                self.queue,
                *args,
            ),
            kwargs=kwargs,
        )
        self.server_process.start()

    def add(self, step: int, *args):
        self.queue.put({"step": step, "args": args})


class SummaryWriterWrapper(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add(self, step: int, *args):
        for data in args:
            inp_args = [data.name, data.value]
            inp_kwargs = {"global_step": step}
            if data.type == "scalar":
                self.add_scalar(*inp_args, **inp_kwargs)
            elif data.type == "image":
                self.add_image(*inp_args, **inp_kwargs)
            elif data.type == "mesh":
                self.add_mesh(*inp_args, **inp_kwargs)
            elif data.type == "histogram":
                self.add_histogram(*inp_args, **inp_kwargs)
            elif data.type == "figure":
                self.add_figure(*inp_args, **inp_kwargs)
