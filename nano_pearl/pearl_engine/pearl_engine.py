import atexit
import time
from dataclasses import fields
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import pickle
import os
import torch
from nano_pearl.pearl_config import PEARLConfig
from nano_pearl.pearl_engine.pearl_model_runner import DraftModelRunner, TargetModelRunner
from nano_pearl.utils.pearl_logger import logger
from multiprocessing.synchronize import Event
from nano_pearl.pearl_engine.sequence import Sequence
from nano_pearl.layers.sampler import SamplingParams


class Controller:
    def __init__(self, config: PEARLConfig, control_event: Event):
        self.config = config
        self.draft_event = []
        self.target_event = []
        self.control_event = control_event
        self.draft_shm = self._create_shm(config.draft_config.group_name)
        self.target_shm = self._create_shm(config.target_config.group_name)

    def _create_shm(self, name: str, size: int = 2**20) -> SharedMemory:
        try:
            return SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            logger.warning("Shared memory '%s' already exists; cleaning it up.", name)
            try:
                existing = SharedMemory(name=name, create=False)
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                pass
            return SharedMemory(name=name, create=True, size=size)

    def add_event(self, rank, event):
        if rank in self.config.draft_config.ranks:
            self.draft_event.append(event)
        else:
            self.target_event.append(event)

    def write_draft_shm(self, method_name, *args):
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.draft_shm.buf[0:4] = n.to_bytes(4, "little")
        self.draft_shm.buf[4:n+4] = data
        for event in self.draft_event:
            event.set()
        
    def write_target_shm(self, method_name, *args):
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.target_shm.buf[0:4] = n.to_bytes(4, "little")
        self.target_shm.buf[4:n+4] = data
        for event in self.target_event:
            event.set()
    
    def read_output(self):
        n = int.from_bytes(self.target_shm.buf[0:4], "little")
        data = self.target_shm.buf[4:n+4]
        output, elapsed_time = pickle.loads(data)
        return output, elapsed_time

    def read_stream_output(self):
        n = int.from_bytes(self.target_shm.buf[0:4], "little")
        data = self.target_shm.buf[4:n+4]
        output, done = pickle.loads(data)
        return output, done


class PEARLEngine:    
    def __init__(self, config: PEARLConfig):
        self.config = config
        self.ps = []
        self.control_timeout_s = float(
            os.getenv("NANO_PEARL_CONTROL_TIMEOUT_S", "120")
        )
        
        ctx = mp.get_context("spawn")
        # the control event is used to wait for the sub-processes to be ready
        self.control_event = ctx.Event()
        self.control_ticket = ctx.Value("i", 0)
        self.controller = Controller(config, self.control_event)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.draft_config.model, use_fast=True)
        config.eos = self.config.draft_config.eos
        logger.info(f"[Main Process] EOS token id: {config.eos}, EOS tokens: {self.tokenizer.decode(config.eos)}")   

        init_ticket = self._next_control_ticket()
        for i in range(config.world_size):
            event = ctx.Event()
            process = ctx.Process(
                target=DraftModelRunner if i in config.draft_config.ranks else TargetModelRunner,
                args=(config, i, event, self.control_event, self.control_ticket),
            )
            process.daemon = True        
            process.start()
            self.ps.append(process)
            self.controller.add_event(i, event)
        
        # wait for the initialization of the draft and target TP models
        logger.info("[Main Process] Waiting for the initialization of the draft and target TP models...", color="red")
        self._wait_for_control_event("init", expected=init_ticket)
        
        atexit.register(self.exit)

    def _maybe_sort_output(self, output):
        if len(output) > 1:
            last = output[0][0]
            for seq_id, _ in output[1:]:
                if seq_id < last:
                    return sorted(output, key=lambda x: x[0])
                last = seq_id
        return output

    def _next_control_ticket(self):
        with self.control_ticket.get_lock():
            return self.control_ticket.value + 1

    def _wait_for_control_event(self, action: str, expected: int):
        deadline = time.monotonic() + self.control_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"PEARLEngine timed out waiting for '{action}' after {self.control_timeout_s}s"
                )
            if not self.control_event.wait(remaining):
                raise TimeoutError(
                    f"PEARLEngine timed out waiting for '{action}' after {self.control_timeout_s}s"
                )
            with self.control_ticket.get_lock():
                current = self.control_ticket.value
            if current >= expected:
                self.control_event.clear()
                return
            # stale signal; clear and keep waiting for the expected ticket.
            self.control_event.clear()

    def log(self, content: str):
        logger.info(f"[Main Process] Running log function, waiting for the sub-processes", color="red")
        expected = self._next_control_ticket()
        self.controller.write_draft_shm("log", content)
        self.controller.write_target_shm("log", content)
        self._wait_for_control_event("log", expected=expected)

    def run_model(self, seqs: list[Sequence], is_prefill: bool):        
        expected = self._next_control_ticket()
        self.controller.write_draft_shm("run_model", seqs, is_prefill)
        self.controller.write_target_shm("run_model", seqs, is_prefill)
        self._wait_for_control_event("run_model", expected=expected)

    def exit(self):
        self.controller.write_draft_shm("exit")
        self.controller.write_target_shm("exit")
        for p in self.ps:
            p.join()                   
        self.controller.draft_shm.close()
        self.controller.target_shm.close()
        self.controller.draft_shm.unlink()
        self.controller.target_shm.unlink()
        

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        expected = self._next_control_ticket()
        if isinstance(prompt, str):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        seq_id = seq.seq_id
        self.controller.write_draft_shm("add_request", seq)
        self.controller.write_target_shm("add_request", seq)
        self._wait_for_control_event("add_request", expected=expected)
        return seq_id

    def cancel_request(self, seq_id: int):
        expected = self._next_control_ticket()
        self.controller.write_draft_shm("cancel_request", seq_id)
        self.controller.write_target_shm("cancel_request", seq_id)
        self._wait_for_control_event("cancel_request", expected=expected)
    
    def generate_tokens(self):
        expected = self._next_control_ticket()
        self.controller.write_draft_shm("pearl_generate")
        self.controller.write_target_shm("pearl_generate")
        self._wait_for_control_event("pearl_generate", expected=expected)

        output, time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        return output, time

    def stream_generate(self):
        expected = self._next_control_ticket()
        self.controller.write_draft_shm("pearl_stream_generate")
        self.controller.write_target_shm("pearl_stream_generate")

        while True:
            self._wait_for_control_event("pearl_stream_generate", expected=expected)
            output, done = self.controller.read_stream_output()
            output = self._maybe_sort_output(output)
            yield output, done
            if done:
                break
            expected = self._next_control_ticket()

    def stream_generate_step(self):
        expected = self._next_control_ticket()
        self.controller.write_draft_shm("pearl_stream_step")
        self.controller.write_target_shm("pearl_stream_step")
        self._wait_for_control_event("pearl_stream_step", expected=expected)
        output, done = self.controller.read_stream_output()
        output = self._maybe_sort_output(output)
        return output, done

    def stream_generate_steps(self, steps: int):
        expected = self._next_control_ticket()
        self.controller.write_draft_shm("pearl_stream_steps", steps)
        self.controller.write_target_shm("pearl_stream_steps", steps)
        self._wait_for_control_event("pearl_stream_steps", expected=expected)
        output, done = self.controller.read_stream_output()
        output = self._maybe_sort_output(output)
        return output, done

    def generate(self):
        output, time = self.generate_tokens()
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]
        
        return output_text, num_tokens, num_acc_tokens, time

    def AR_generate(self):
        """Only use target model for Auto-Regressive generation."""
        expected = self._next_control_ticket()
        self.controller.write_draft_shm("parallel_generate")
        self.controller.write_target_shm("parallel_generate")
        self._wait_for_control_event("parallel_generate", expected=expected)

        output, time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, _ = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]

        return output_text, num_tokens, None, time
    
    def bench_generate(self, num_pearl_steps: int = 100):
        expected = self._next_control_ticket()
        self.controller.write_draft_shm("pearl_bench_generate", num_pearl_steps)
        self.controller.write_target_shm("pearl_bench_generate", num_pearl_steps)
        self._wait_for_control_event("pearl_bench_generate", expected=expected)

        output, time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]

        return output_text, num_tokens, num_acc_tokens, time
