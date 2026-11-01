# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MoRI checkpoint engine: actor -> rollout weight sync over MoRI-IO (https://github.com/ROCm/mori) RDMA WRITE.

Topology: P stripes, P = min(num_chains, actor world size). Actor rank c leads stripe c and sends the tensors assigned
to it; each tensor goes to the stripe with the fewest bytes so far (every actor rank iterates the same generator in the
same order, as it must for the trainer's collectives, so they all compute the same assignment without talking).
Rollout workers are grouped by node. On every rollout node one GPU holds stripe c (local GPU c % n); the holders form a
chain across nodes (actor c -> holder on node 1 -> holder on node 2 ...), so each rollout node receives one copy of the
model spread over its NICs, and each holder copies its stripe to the other GPUs of its node, over XGMI when available
(MoRI routes same-node GPU pairs to XGMI and the rest to RDMA). num_chains=1 is a single stripe. With export="shard"
stripe r instead carries trainer rank r's own shards of every tensor, and receivers put each full tensor together
from the pieces of all stripes.

Every worker allocates ONE device region of num_slots * bucket_size bytes, split into num_slots slots (2: double
buffering) per stripe, each bucket_size / num_chains bytes, and registers it once (first `prepare`). The region, the
remote engine registrations, QPs and ZMQ sockets are kept across syncs; `finalize` only forgets the positions.
Data moves with one-sided WRITE from my slot (c, j) into the same slot of each next hop. Control messages (ZMQ):
  * prev -> me      {"t": "bucket", c, k, slot, uid, metas, used, is_last}  bucket k of stripe c is in my slot (c, slot)
  * me   -> prev    {"t": "free", c, slot, from}                            my slot (c, slot) may be overwritten
  * me   -> all     {"t": "abort", reason}                                  I failed; fail now instead of timing out
Bucket k of a stripe always uses slot k % num_slots. A receiver checks MoRI's inbound completion notification for
(prev, uid) before using a bucket, writes it on to all its next hops, yields its tensors, and frees the slot once every
forward write and local consumption are done. Tensors are merged per stripe and handed to the consumer one at a time; a
stripe only moves past a tensor once the consumer has asked for the next one. No worker synchronizes the whole device:
each bucket's write waits for an event recorded after its packing copies, so packing overlaps transfers, and a
receiver frees a slot once an event recorded after the consumer's reads of it fired (consumers read on the current
stream, as verl's server adapters do), while it already moves on to the next bucket.

Requirements found on MI355X + Pollara (ionic) with MoRI 1.2.4:
  * Buckets must not come from PyTorch expandable segments (VMM): MoRI fails to register them (ibv_reg_mr and
    dma-buf both EINVAL). They are allocated with cupy (plain hipMalloc), as the nccl/nixl engines do.
  * `enable_transfer_chunking` must be on: off halves single-transfer bandwidth (25 vs 47 GB/s per NIC).
  * MoRI picks the NIC closest to the region's GPU (by PCI bus id), so Ray's per-worker device masking is fine.
  * Same-node hops go over XGMI because the engine adds MoRI's XGMI backend (intra_node="xgmi"). XGMI writes raise no
    inbound notification at the target (XgmiBackend::PopInboundTransferStatus), so the receiver of such a hop relies
    on the bucket message being sent only after the write completed; a bounded `WaitFor` on an XGMI transfer sleeps
    to its timeout (mori#873), so transfers are polled.

Args (engine_kwargs["mori"]):
    num_chains (int): Max stripes; the actual count is min(num_chains, actor world size). Default 8.
    num_slots (int): Buckets in flight per stripe and hop (>= 2). More slots hide per-bucket latency (control
        messages, forwarding) at num_slots * bucket_size bytes of device memory per worker. Default 2.
    export (str): Where stripes come from. "full" (default): the trainer's full-tensor generator
        (`get_per_tensor_param`); every rank iterates it and each tensor goes to one stripe. "shard": every trainer
        rank sends its own shards from `get_per_tensor_param_shard()` as stripe `rank` (bf16, no all-gather on the
        trainer; needs num_chains >= trainer world size and shards that are one flat range of the full tensor, e.g.
        FSDP2 Shard(0)); receivers assemble full tensors and split packed MoE experts as the full export does. The
        trainer worker then passes its training engine to `send_weights` (`consumes_training_engine`).
    intra_node (str): "xgmi" (default) adds MoRI's XGMI backend so same-node copies use XGMI; "rdma" keeps them on
        the NIC (loopback).
    node_id (str): Node identity used to group rollout workers. Default: the Ray node id (tests set it).
    qp_per_transfer (int): RDMA QPs per transfer. Default 4.
    num_nics_per_transfer (int): NICs a single transfer may stripe over. Default 1 (the GPU's own NIC).
    enable_chunking (bool): MoRI transfer chunking. Default True.
    host (str): IP for the MoRI control plane and ZMQ. Default: Ray node IP.
    timeout_s (float): Max wait for any single transfer or control message. Default 600.
    device (str): "cuda" (default) or "cpu" (tests only).
    verify (bool | str): Debug checksums, printed as `MORI_VERIFY ...` lines; costs device syncs.
        "send": sender checksums every chunk before packing (one sync per chunk) and checks the packed bucket.
        "recv": sender only checksums each packed bucket (it syncs there anyway); every receiver checks arrival and,
                on a mismatch, re-checks for up to 2 s to tell late visibility from corruption.
        "all" / True: both. Default: env MORI_CKPT_VERIFY ("1" means "all"; unset or "0" means off).
Env MORI_CKPT_STATS=1 prints one `MORI_STATS` line per sync and worker with a time breakdown.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import socket
import sys
import time
import uuid
from types import SimpleNamespace
from typing import AsyncGenerator, Generator

import torch
import zmq
import zmq.asyncio

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)
from verl.workers.rollout.utils import ensure_async_iterator

# MoRI is imported lazily (first prepare()), so importing verl.checkpoint_engine never loads the MoRI/HIP stack;
# only its presence is checked here, which lets verl record a missing dependency as an import error.
if importlib.util.find_spec("mori") is None:
    raise ImportError("the mori checkpoint engine needs MoRI (pip install amd-mori)")

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_ALIGN = 256  # chunk offsets inside a bucket and bucket sizes; keeps every dtype view aligned


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _alloc_region(nbytes: int, device: str) -> tuple[torch.Tensor, object]:
    """Allocate the region outside PyTorch's caching allocator (no expandable segments). Returns (tensor, owner)."""
    if device != "cuda":
        return torch.zeros(nbytes, dtype=torch.uint8, device=device), None
    try:
        from unittest.mock import patch

        with patch("importlib.metadata.distributions", return_value=[]):
            import cupy as cp
        owner = cp.zeros(nbytes, dtype=cp.uint8)
        t = torch.as_tensor(owner, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()  # the zero-fill runs on cupy's stream; finish it before anything else uses the region
        return t, owner
    except Exception as e:  # no cupy, or no __cuda_array_interface__ interop: allocate with expandable segments off
        logger.warning(f"mori: cupy region allocation failed ({e!r}); using torch with expandable segments off")
        conf = os.environ.get("PYTORCH_ALLOC_CONF", "") + os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        try:
            t = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            return t, None
        finally:
            if "expandable_segments:True" in conf:
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")


def _csum(t: torch.Tensor) -> tuple[int, int]:
    """Cheap checksum of a uint8 tensor (sum of all bytes, sum of odd bytes)."""
    return int(torch.sum(t, dtype=torch.int64).item()), int(torch.sum(t[1::2], dtype=torch.int64).item())


def _shard_flat_offset(spec) -> tuple[int, bool]:
    """(element offset of this rank's local shard in the flattened full tensor, whether this rank sends it) for a
    `verl.workers.engine.spec.ShardSpec`. Only shards that are one flat range of the full tensor are supported (only
    dim 0 is cut, e.g. FSDP2 Shard(0)); replicated / unsharded tensors are sent by the first replica only."""
    if spec.place is not None:  # explicit exporter override
        place, contributes = spec.place, spec.contributes
    else:
        from verl.workers.engine.spec import derive_dtensor_placement

        place, contributes, _ = derive_dtensor_placement(spec)
    if isinstance(place, int):
        return place, contributes
    if not place.is_flat_contiguous:
        raise NotImplementedError(
            f"mori export='shard' needs shards that are one flat range of the full tensor (only dim 0 cut); got local "
            f"shape {place.local_shape} at offset {place.global_offset} of {place.full_shape}. Use export='full'."
        )
    return place.flat_offset, contributes


def _enable_debug_output():
    """Opt-in debug lines (MORI_CKPT_STATS, verify) are logged at INFO by this module's logger; make sure they are
    emitted in Ray workers, where logging is usually not configured."""
    if not any(getattr(h, "_mori_debug", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._mori_debug = True
        logger.addHandler(handler)
        logger.propagate = False
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)


def _node_ip() -> str:
    try:
        import ray

        return ray.util.get_node_ip_address().strip("[]")
    except Exception:
        return socket.gethostbyname(socket.gethostname())


@CheckpointEngineRegistry.register("mori")
class MoriCheckpointEngine(CheckpointEngine):
    """MoRI-IO checkpoint engine (RDMA WRITE chains, one persistent registered region). See module docstring.

    Args:
        bucket_size (int): Region size is num_slots * bucket_size per process; each chain gets num_slots slots of
            bucket_size / num_chains bytes.
        is_master (bool): Unused; accepted for signature compatibility with the other engines.
    """

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        num_chains: int = 8,
        num_slots: int = 2,
        qp_per_transfer: int = 4,
        num_nics_per_transfer: int = 1,
        enable_chunking: bool = True,
        host: str | None = None,
        timeout_s: float = 600.0,
        device: str = "cuda",
        verify: bool | str | None = None,
        intra_node: str = "xgmi",
        node_id: str | None = None,
        export: str = "full",
    ):
        self.bucket_size = bucket_size
        self.is_master = is_master
        self.num_chains = max(1, int(num_chains))
        self.chain_bucket = max(_ALIGN, bucket_size // self.num_chains // _ALIGN * _ALIGN)
        assert int(num_slots) >= 2, f"num_slots must be >= 2, got {num_slots}"
        self.num_slots = int(num_slots)
        self.qp_per_transfer = qp_per_transfer
        self.num_nics_per_transfer = num_nics_per_transfer
        self.enable_chunking = enable_chunking
        self.host = host
        self.timeout_s = timeout_s
        self.device = device
        assert intra_node in ("xgmi", "rdma"), f"intra_node must be xgmi or rdma, got {intra_node!r}"
        self.intra_node = intra_node
        self.node_id = node_id
        assert export in ("full", "shard"), f"export must be full or shard, got {export!r}"
        self.export = export
        self._unfuse_moe_params = None
        if export == "shard":
            # imported now rather than on the first received tensor: importing verl's engine package takes seconds,
            # which would land in the first sync
            from verl.workers.engine.fsdp.utils import unfuse_moe_params

            self._unfuse_moe_params = unfuse_moe_params
        if verify is None:
            verify = os.environ.get("MORI_CKPT_VERIFY", "0")
        verify = {"1": "all", "0": "", True: "all", False: "", None: ""}.get(verify, verify)
        assert verify in ("", "all", "send", "recv"), f"bad verify mode {verify!r}"
        assert not (verify and export == "shard"), "verify is only implemented for export='full'"
        self.verify = verify
        self._v_send = verify in ("all", "send")  # per-chunk checksums at the sender
        self._v_recv = verify in ("all", "recv")  # arrival checks at receivers
        self._stats_on = os.environ.get("MORI_CKPT_STATS", "0") == "1"
        if self._stats_on or self.verify:
            _enable_debug_output()
        self._chain_rank = None  # last position, kept for logs after finalize

        # created once, on first prepare(), and kept across syncs
        self._engine = None  # mori.io.IOEngine
        self._region: torch.Tensor | None = None
        self._owner = None
        self._mem = None  # mori.io.MemoryDesc of my region
        self._key: str | None = None
        self._zctx = None
        self._inbox = None
        self._inbox_addr: str | None = None
        self._push: dict[str, object] = {}  # inbox address -> PUSH socket
        self._peers: dict = {}  # key -> mori.io.EngineDesc of remote engines registered with MoRI
        self._warm: set[str] = set()  # next-hop keys whose QPs/MRs are already built
        self._seq = 0  # sync counter; advances in lockstep on every worker

        # per sync (set by init_process_group, cleared by finalize)
        self.rank: int | None = None
        self._P = 0
        self._roles: dict[int, SimpleNamespace] = {}  # chain -> SimpleNamespace(prev, next)

    @property
    def consumes_training_engine(self) -> bool:
        """With export="shard" the trainer hands its training engine to `send_weights` (not a full-tensor generator):
        every rank sends its own shards from `get_per_tensor_param_shard()`, so the trainer never all-gathers."""
        return self.export == "shard"

    def _off(self, c: int, j: int) -> int:
        return (self.num_slots * c + j) * self.chain_bucket

    def _slot(self, c: int, j: int) -> torch.Tensor:
        o = self._off(c, j)
        return self._region[o : o + self.chain_bucket]

    # ------------------------------------------------------------------ setup
    def _setup(self):
        from mori.io import BackendType, IOEngine, IOEngineConfig, RdmaBackendConfig

        ip = self.host or _node_ip()
        key = f"verl-mori-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        engine = IOEngine(key=key, config=IOEngineConfig(host=ip, port=0))
        engine.create_backend(
            BackendType.RDMA,
            RdmaBackendConfig(
                qp_per_transfer=self.qp_per_transfer,
                enable_notification=True,
                enable_transfer_chunking=self.enable_chunking,
                num_nics_per_transfer=self.num_nics_per_transfer,
            ),
        )
        self._xgmi = False
        if self.intra_node == "xgmi":
            # with both backends MoRI sends same-node GPU pairs over XGMI and everything else over RDMA
            try:
                from mori.io import XgmiBackendConfig

                engine.create_backend(BackendType.XGMI, XgmiBackendConfig())
                self._xgmi = True
            except Exception as e:
                logger.warning(f"mori: XGMI backend unavailable ({e!r}); intra-node copies go over RDMA")
        self._region, self._owner = _alloc_region(self.num_slots * self.num_chains * self.chain_bucket, self.device)
        self._mem = engine.register_torch_tensor(self._region)
        self._zctx = zmq.asyncio.Context()
        # never block on close: with the default (infinite) linger, garbage-collecting the context waits for every
        # undelivered message, e.g. an abort to a peer that is gone, and hangs the process at exit
        self._zctx.setsockopt(zmq.LINGER, 0)
        self._inbox = self._zctx.socket(zmq.PULL)
        if ":" in ip:
            self._inbox.setsockopt(zmq.IPV6, 1)
            port = self._inbox.bind_to_random_port("tcp://[::]")
            self._inbox_addr = f"tcp://[{ip}]:{port}"
        else:
            port = self._inbox.bind_to_random_port("tcp://0.0.0.0")
            self._inbox_addr = f"tcp://{ip}:{port}"
        self._engine, self._key = engine, key
        logger.info(
            f"mori engine {key} ready: inbox {self._inbox_addr}, {self.num_chains} chains x {self.num_slots} x "
            f"{self.chain_bucket >> 20} MiB"
        )

    def prepare(self) -> dict:
        """Create the MoRI engine and register the region on first use; return this worker's metadata."""
        if self._engine is None:
            self._setup()
        return {
            "engine_desc": self._engine.get_engine_desc().pack(),
            "region": self._mem.pack(),
            "inbox": self._inbox_addr,
            "num_chains": self.num_chains,
            "chain_bucket": self.chain_bucket,
            "num_slots": self.num_slots,
            "export": self.export,
            "node": self._node(),
            "xgmi": self._xgmi,
        }

    def _node(self) -> str:
        if self.node_id is not None:
            return self.node_id
        try:
            import ray

            return ray.get_runtime_context().get_node_id()
        except Exception:
            return socket.gethostname()

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        """Stripe c (one per trainer rank, P = min(num_chains, trainer ranks)) is carried across nodes by one rollout
        GPU per node, its holder: local GPU c % n on every rollout node, so each node takes one copy of the model spread
        over its NICs. The holders form a chain trainer c -> node 1 -> node 2 -> ... (node order rotated by c), and
        each holder also copies the stripe to every other GPU of its node (over XGMI when available)."""
        A, R = metadata[:actor_wg_world_size], metadata[actor_wg_world_size:]
        N = rollout_world_size
        assert len(R) == N and N > 0
        assert len({(m["chain_bucket"], m["num_slots"], m["export"]) for m in metadata}) == 1, (
            "all workers need the same bucket size, num_chains, num_slots and export"
        )
        if metadata[0]["export"] == "shard":  # every trainer rank sends its own shards: one stripe per rank
            P = actor_wg_world_size
            assert min(m["num_chains"] for m in metadata) >= P, (
                f"export='shard' needs num_chains >= trainer world size ({P}): each rank's shards form a stripe"
            )
        else:
            P = min(min(m["num_chains"] for m in metadata), actor_wg_world_size)
        nodes: dict[str, list[int]] = {}
        for r, m in enumerate(R):
            nodes.setdefault(m["node"], []).append(r)
        node_list = list(nodes.values())

        def holders(c: int) -> list[int]:  # rollout index holding stripe c on each node, in chain order
            order = node_list[c % len(node_list) :] + node_list[: c % len(node_list)]
            return [members[c % len(members)] for members in order]

        actor_roles = [[] for _ in range(actor_wg_world_size)]
        rollout_roles = [[] for _ in range(N)]
        for c in range(P):
            hs = holders(c)
            actor_roles[c].append({"c": c, "prev": None, "nexts": [R[hs[0]]]})
            for i, h in enumerate(hs):
                local = next(members for members in node_list if h in members)
                nexts = ([R[hs[i + 1]]] if i + 1 < len(hs) else []) + [R[x] for x in local if x != h]
                rollout_roles[h].append({"c": c, "prev": A[c] if i == 0 else R[hs[i - 1]], "nexts": nexts})
                for x in local:
                    if x != h:
                        rollout_roles[x].append({"c": c, "prev": R[h], "nexts": []})
        actor_wg_kwargs = {
            "rank": [a if a < P else -1 for a in range(actor_wg_world_size)],
            "num_chains": [P] * actor_wg_world_size,
            "chain_roles": actor_roles,
        }
        rollout_kwargs = {
            "rank": [P + r for r in range(N)],  # log label only; trainer stripe leaders are 0..P-1
            "num_chains": [P] * N,
            "chain_roles": rollout_roles,
        }
        return actor_wg_kwargs, rollout_kwargs

    def _connect(self, meta: dict | None):
        from mori.io import EngineDesc, MemoryDesc

        if meta is None:
            return None
        desc = EngineDesc.unpack(meta["engine_desc"])
        if desc.key not in self._peers:
            self._engine.register_remote_engine(desc)
            self._peers[desc.key] = desc
        sock = self._push.get(meta["inbox"])
        if sock is None:
            sock = self._zctx.socket(zmq.PUSH)
            if meta["inbox"].startswith("tcp://["):
                sock.setsockopt(zmq.IPV6, 1)
            sock.connect(meta["inbox"])
            self._push[meta["inbox"]] = sock
        # MoRI routes a write over XGMI when both engines have the XGMI backend and sit on the same node; XGMI writes
        # raise no target-side notification (XgmiBackend::PopInboundTransferStatus always returns false), so the
        # receiver of such a hop relies on the sender announcing the bucket only after its write completed.
        xgmi_hop = self._xgmi and meta.get("xgmi", False) and meta.get("node") == self._node()
        return SimpleNamespace(key=desc.key, sock=sock, region=MemoryDesc.unpack(meta["region"]), notifies=not xgmi_hop)

    def init_process_group(self, rank: int, num_chains: int, chain_roles: list[dict]):
        """Connect to my previous hop and next hops in every stripe I am in (connections are cached across syncs)."""
        self.rank, self._chain_rank, self._P = rank, rank, num_chains
        self._roles = {}
        for role in chain_roles:
            self._roles[role["c"]] = SimpleNamespace(
                prev=self._connect(role["prev"]), nexts=[self._connect(m) for m in role["nexts"]]
            )
        # Build QPs / IPC mappings and fetch the remote MR now (MoRI does it lazily on the first write), so the first
        # sync does not pay for it: one tiny write per next hop into its slot (c, 0), overwritten before it is read.
        for c, ro in self._roles.items():
            for nxt in ro.nexts:
                if nxt.key in self._warm:
                    continue
                st = self._engine.write(
                    self._mem,
                    self._off(c, 0),
                    nxt.region,
                    self._off(c, 0),
                    _ALIGN,
                    self._engine.allocate_transfer_uid(),
                )
                t0 = time.time()
                while st.InProgress():  # poll: a bounded WaitFor sleeps to its timeout on XGMI (mori#873)
                    if time.time() - t0 > self.timeout_s:
                        raise TimeoutError(f"mori warm-up write to {nxt.key} timed out")
                    time.sleep(0.0002)
                if not st.Succeeded():
                    raise RuntimeError(f"mori warm-up write to {nxt.key} failed: {st.Message()}")
                self._warm.add(nxt.key)

    def finalize(self):
        """Keep the region, registrations, QPs and sockets for the next sync; only forget this sync's positions."""
        self.rank = None
        self._roles = {}

    def close(self):
        """Release sockets, the ZMQ context and the MoRI registration. Not part of the per-sync API (`finalize` keeps
        everything for the next sync); call it when the engine is torn down for good, e.g. in tests."""
        for sock in self._push.values():
            sock.close(linger=0)
        self._push.clear()
        if self._inbox is not None:
            self._inbox.close(linger=0)
            self._inbox = None
        if self._zctx is not None:
            self._zctx.term()
            self._zctx = None
        if self._engine is not None and self._mem is not None:
            try:
                self._engine.deregister_memory(self._mem)
            except Exception as e:  # best effort at teardown
                logger.warning(f"mori: deregister_memory failed at close: {e!r}")
        self._engine = self._mem = self._region = self._owner = None
        self._peers.clear()
        self._warm.clear()

    # ------------------------------------------------------------------ control plane
    def _start_pump(self, seq: int):
        """Single reader of the inbox: bucket messages -> per-chain queues, free messages -> per-slot credits."""
        self._buckets = {c: asyncio.Queue() for c in self._roles}
        self._abort: str | None = None
        self._credit = {
            (c, j, n.key): asyncio.Event()
            for c, ro in self._roles.items()
            for n in ro.nexts
            for j in range(self.num_slots)
        }
        for e in self._credit.values():
            e.set()  # every next hop's slots are free at the start of a sync

        async def pump():
            while True:
                msg = await self._inbox.recv_pyobj()
                if msg.get("seq") != seq:
                    logger.warning(f"mori: dropping message from sync {msg.get('seq')} during sync {seq}: {msg['t']}")
                    continue
                if msg["t"] == "bucket":
                    self._buckets[msg["c"]].put_nowait(msg)
                elif msg["t"] == "free":
                    self._credit[(msg["c"], msg["slot"], msg["from"])].set()
                elif msg["t"] == "abort":  # wake every waiter; they re-check self._abort
                    self._abort = msg["reason"]
                    for e in self._credit.values():
                        e.set()
                    for q in self._buckets.values():
                        q.put_nowait(msg)

        self._pump = asyncio.ensure_future(pump())

    def _check_abort(self):
        if self._abort is not None:
            raise RuntimeError(f"mori: aborted by a peer in the chain: {self._abort}")

    async def _notify_abort(self, seq: int, err: BaseException):
        """Best effort: tell every neighbour this sync failed so it fails now instead of after timeout_s."""
        if isinstance(err, RuntimeError) and "aborted by a peer" in str(err):
            reason = self._abort  # pass the original reason along
        else:
            reason = f"{self._key} (rank {self._chain_rank}): {type(err).__name__}: {err}"
        peers = {p.sock for ro in self._roles.values() for p in (ro.prev, *ro.nexts) if p is not None}
        for sock in peers:
            try:
                await sock.send_pyobj({"t": "abort", "seq": seq, "reason": reason}, flags=zmq.NOBLOCK)
            except Exception:
                pass

    @staticmethod
    async def _reap(tasks):
        """Cancel unfinished tasks and collect every task's exception so none is reported as 'never retrieved'."""
        for t in tasks:
            if t is None:
                continue
            if not t.done():
                t.cancel()
            try:
                await t
            except BaseException:
                pass

    async def _stop_pump(self):
        self._pump.cancel()
        try:
            await self._pump
        except asyncio.CancelledError:
            pass

    async def _timed(self, aw, what: str):
        try:
            return await asyncio.wait_for(aw, self.timeout_s)
        except asyncio.TimeoutError:
            failure = self._engine.pop_peer_failure() if self._engine is not None else None
            raise TimeoutError(
                f"mori: timed out after {self.timeout_s}s waiting for {what}; peer failure: {failure}"
            ) from None

    async def _wait_status(self, st, what: str):
        async def poll():
            while st.InProgress():
                await asyncio.sleep(0.0002)

        await self._timed(poll(), what)
        if not st.Succeeded():
            raise RuntimeError(f"mori: {what} failed: {st.Message()}")

    async def _wait_inbound(self, prev_key: str, uid: int, what: str):
        async def poll():
            while True:
                st = self._engine.pop_inbound_transfer_status(prev_key, uid)
                if st is not None:
                    return st
                await asyncio.sleep(0.0002)

        st = await self._timed(poll(), what)
        if st.Failed():
            raise RuntimeError(f"mori: {what} reported failure: {st.Message()}")

    async def _write_next(self, seq, c, k, j, metas, used, is_last, steps, sums=None, ready=None, model_type=None):
        """Write my slot (c, j) into the same slot of every next hop (in parallel) and announce it to each.

        `ready` is a device event recorded after the packing copies; the NIC must not read the slot before it."""
        if ready is not None:
            t0 = time.perf_counter()
            while not ready.query():
                await asyncio.sleep(0.0001)
            self._acc["ready_wait"] += time.perf_counter() - t0
        msg = {
            "t": "bucket",
            "seq": seq,
            "c": c,
            "k": k,
            "slot": j,
            "metas": metas,
            "used": used,
            "is_last": is_last,
            "global_steps": steps,
            "sums": sums,
            "model_type": model_type,  # export="shard": receivers unfuse MoE experts as the trainer's full export does
        }

        async def one(nxt):
            t0 = time.perf_counter()
            await self._timed(
                self._credit[(c, j, nxt.key)].wait(), f"{nxt.key} to free slot {j} of stripe {c} (bucket {k})"
            )
            self._check_abort()
            self._credit[(c, j, nxt.key)].clear()
            t1 = time.perf_counter()
            uid = None
            if used > 0:
                uid = self._engine.allocate_transfer_uid()
                st = self._engine.write(self._mem, self._off(c, j), nxt.region, self._off(c, j), used, uid)
                await self._wait_status(st, f"write of bucket {k} of stripe {c} ({used} B) to {nxt.key}")
            await nxt.sock.send_pyobj({**msg, "uid": uid, "notified": nxt.notifies})
            # summed over next hops, which run concurrently
            self._acc["credit_wait"] += t1 - t0
            self._acc["write"] += time.perf_counter() - t1

        tasks = [asyncio.ensure_future(one(n)) for n in self._roles[c].nexts]
        try:
            await asyncio.gather(*tasks)
        finally:
            await self._reap(tasks)

    async def _drain_credits(self, c: int, what: str):
        for n in self._roles[c].nexts:
            for j in range(self.num_slots):
                await self._timed(
                    self._credit[(c, j, n.key)].wait(), f"{what}: {n.key} to release slot {j} of stripe {c}"
                )
        self._check_abort()

    async def _release(self, seq, c, j, forward, consumed, st):
        """Free my slot (c, j) at my previous hop once `forward` (my writes of it to my next hops) completed and the
        `consumed` event (recorded after the consumer's reads of it) fired."""
        t0 = time.perf_counter()
        if forward is not None:
            await forward
        t1 = time.perf_counter()
        if consumed is not None:
            while not consumed.query():
                await asyncio.sleep(0.0001)
        self._check_abort()
        await self._roles[c].prev.sock.send_pyobj({"t": "free", "seq": seq, "c": c, "slot": j, "from": self._key})
        st["wait_forward"] += t1 - t0
        st["free"] += time.perf_counter() - t1

    def _mark_ready(self):
        """Event after the copies queued so far on the current stream (the NIC may read the slot once it fires)."""
        if self.device != "cuda":
            return None
        ev = torch.cuda.Event()
        ev.record()
        return ev

    def _sync_device(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _stats(self, side, dt, buckets, nbytes, st):
        parts = " ".join(f"{k}={v:.3f}" for k, v in st.items())
        logger.info(
            f"MORI_STATS {side} rank={self._chain_rank} pid={os.getpid()} total={dt:.3f} chains={self._P} "
            f"buckets={buckets} GiB={nbytes / 2**30:.2f} {parts}"
        )

    # ------------------------------------------------------------------ verification (debug)
    def _vlog(self, what: str, **kv):
        parts = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info(f"MORI_VERIFY {what} rank={self._chain_rank} pid={os.getpid()} {parts}")

    def _verify_packed(self, c, k, j, metas, srcsums, used):
        """Sender: each packed region must equal its source chunk; returns the checksums to send along."""
        buf = self._slot(c, j)
        if not self._v_send:  # recv mode: per-tensor sums of the packed bytes, so receivers can name bad tensors
            return {
                "bucket": _csum(buf[:used]),
                "tensors": [_csum(buf[m.offset : m.offset + m.chunk_size]) for m in metas],
            }
        bad = [
            m.name for m, s in zip(metas, srcsums, strict=True) if _csum(buf[m.offset : m.offset + m.chunk_size]) != s
        ]
        if bad:
            self._vlog("pack_mismatch", c=c, k=k, n=len(bad), names=bad[:5])
            raise RuntimeError(f"mori verify: bucket {k} of chain {c} packed bytes differ from source for {bad[:5]}")
        return {"bucket": _csum(buf[:used]), "tensors": list(srcsums)}

    async def _verify_arrival(self, c, k, j, msg, prev_key):
        """Receiver: bucket must match the sender's checksum; if not, poll to tell late visibility from corruption."""
        buf, want = self._slot(c, j), tuple(msg["sums"]["bucket"])
        if _csum(buf[: msg["used"]]) == want:
            self._v_ok = getattr(self, "_v_ok", 0) + 1
            return
        t0 = time.time()
        while time.time() - t0 < 2.0:
            await asyncio.sleep(0.001)
            self._sync_device()
            if _csum(buf[: msg["used"]]) == want:
                self._vlog("late_visibility", c=c, k=k, ms=round((time.time() - t0) * 1000, 2), used=msg["used"])
                return
        bad = [
            m.name
            for m, s in zip(msg["metas"], msg["sums"]["tensors"], strict=True)
            if _csum(buf[m.offset : m.offset + m.chunk_size]) != tuple(s)
        ]
        zeros = [
            m.name
            for m in msg["metas"]
            if m.name in bad[:5] and int(torch.count_nonzero(buf[m.offset : m.offset + m.chunk_size]).item()) == 0
        ]
        self._vlog(
            "corrupt",
            c=c,
            k=k,
            used=msg["used"],
            n_bad=len(bad),
            n_tensors=len(msg["metas"]),
            names=bad[:5],
            all_zero=zeros,
        )
        raise RuntimeError(
            f"mori verify: bucket {k} of chain {c} from {prev_key} corrupt for {len(bad)} tensors, e.g. {bad[:3]}"
        )

    # ------------------------------------------------------------------ data plane
    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """export="full": `weights` yields full tensors; every actor rank must iterate them (the trainer's collectives
        run inside the generator), and the leader of stripe c packs and pushes the tensors assigned to stripe c.
        export="shard": `weights` is the training engine (or a generator of (name, local shard, ShardSpec)); every
        rank pushes its own shards as stripe `rank`, without any collective."""
        assert self.rank is not None, "init_process_group was not called"
        if self.export == "shard":
            gen, self._send_model_type = weights, None
            if hasattr(weights, "get_per_tensor_param_shard"):  # a verl training engine
                gen, _ = weights.get_per_tensor_param_shard()
                hf_config = getattr(getattr(weights, "model_config", None), "hf_config", None)
                self._send_model_type = getattr(hf_config, "model_type", None)
            pieces = self._shard_pieces(gen)
        elif not self._roles:  # not leading a stripe: only drive the generator (it runs the trainer's collectives)
            async for _ in ensure_async_iterator(weights):
                pass
            return
        else:
            self._send_model_type = None
            pieces = self._stripe_pieces(weights)
        await self._push_stripe(pieces, global_steps)

    async def _stripe_pieces(self, weights) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
        """export="full": chunks of the tensors of my stripe. Each tensor goes to the least-loaded stripe so far;
        every actor rank sees the same tensors in the same order, so all compute the same assignment."""
        (c,) = self._roles.keys()
        loads, mine = [0] * self._P, False
        async for tensor_meta, chunk in split_weight_chunks(weights, self.chain_bucket):
            if tensor_meta.chunk_offset == 0:  # a new tensor
                to = min(range(self._P), key=loads.__getitem__)
                loads[to] += tensor_meta.shape.numel() * tensor_meta.dtype.itemsize
                mine = to == c
            if mine:
                yield tensor_meta, chunk

    async def _shard_pieces(self, gen) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
        """export="shard": chunks of my own local shards. TensorMeta.shape / dtype describe the full tensor and
        chunk_offset is the chunk's byte offset in it, so receivers assemble full tensors from every rank's pieces."""
        cb = self.chain_bucket
        async for name, local, spec in ensure_async_iterator(gen):
            full_shape = torch.Size(spec.full_shape)
            itemsize = local.dtype.itemsize
            if full_shape.numel() == 0:  # nothing to send, but the receivers must still get the (empty) tensor
                if self.rank == 0:
                    yield TensorMeta(name, full_shape, local.dtype, 0, 0, None), local.reshape(-1).view(torch.uint8)
                continue
            first, contributes = _shard_flat_offset(spec)
            if not contributes or local.numel() == 0:
                continue
            buf = local.contiguous().reshape(-1).view(torch.uint8)
            pos = 0
            while pos < buf.numel():
                n = min(cb, buf.numel() - pos)
                yield TensorMeta(name, full_shape, local.dtype, first * itemsize + pos, n, None), buf[pos : pos + n]
                pos += n

    async def _push_stripe(self, pieces, global_steps):
        """Pack the pieces of my stripe into my slots and push each full bucket down the stripe."""
        (c,) = self._roles.keys()
        self._seq += 1
        seq = self._seq
        self._start_pump(seq)
        cb = self.chain_bucket
        start, nbytes, k, j, offset = time.time(), 0, 0, 0, 0
        metas: list[TensorMeta] = []
        srcsums: list = []  # verify: checksum of each chunk before it was packed
        writing: list[asyncio.Future | None] = [None] * self.num_slots  # write task reading from my slot (c, j)
        st = dict.fromkeys(["first", "export", "wait_slot", "tail"], 0.0)
        self._acc = dict.fromkeys(["ready_wait", "credit_wait", "write"], 0.0)
        mt = self._send_model_type
        chunks = pieces.__aiter__()
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    tensor_meta, chunk = await chunks.__anext__()
                except StopAsyncIteration:
                    break
                st["export"] += time.perf_counter() - t0
                if not st["first"]:  # until the trainer's generator produced my first tensor
                    st["first"] = max(time.time() - start, 1e-9)
                at = _align(offset)
                if at + tensor_meta.chunk_size > cb:
                    # the NIC may only read the slot once the packing copies have run; the write task waits for this
                    # event, so packing the next bucket overlaps with the copy and the transfer of this one
                    if self.verify:
                        self._sync_device()
                    sums = self._verify_packed(c, k, j, metas, srcsums, offset) if self.verify else None
                    writing[j] = asyncio.ensure_future(
                        self._write_next(seq, c, k, j, metas, offset, False, global_steps, sums, self._mark_ready(), mt)
                    )
                    nbytes += offset
                    k, j, offset, at, metas, srcsums = k + 1, (k + 1) % self.num_slots, 0, 0, [], []
                    if writing[j] is not None:  # my slot (c, j) is still being read by an earlier write
                        t0 = time.perf_counter()
                        await writing[j]
                        st["wait_slot"] += time.perf_counter() - t0
                tensor_meta.offset = at
                metas.append(tensor_meta)
                if self._v_send:
                    self._sync_device()
                    srcsums.append(_csum(chunk))
                self._slot(c, j)[at : at + tensor_meta.chunk_size].copy_(chunk, non_blocking=True)
                offset = at + tensor_meta.chunk_size
            t0 = time.perf_counter()
            if self.verify:
                self._sync_device()
            sums = self._verify_packed(c, k, j, metas, srcsums, offset) if self.verify else None
            writing[j] = asyncio.ensure_future(
                self._write_next(seq, c, k, j, metas, offset, True, global_steps, sums, self._mark_ready(), mt)
            )
            nbytes += offset
            for w in writing:
                if w is not None:
                    await w
            await self._drain_credits(c, "send_weights")
            st["tail"] += time.perf_counter() - t0
        except BaseException as e:
            await self._notify_abort(seq, e)
            raise
        finally:
            await self._reap(writing)
            await self._stop_pump()
        dt = time.time() - start
        if self._stats_on:
            self._stats("send", dt, k + 1, nbytes, {**st, **self._acc})
        logger.info(
            f"mori chain {c}: sent {k + 1} buckets, {nbytes / 2**30:.2f} GiB in {dt:.2f}s "
            f"({nbytes / max(dt, 1e-9) / 2**30:.2f} GiB/s)"
        )

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive every chain, forward each bucket down its chain, and yield (name, tensor) one at a time."""
        assert self.rank is not None and self._roles, "init_process_group was not called"
        self._seq += 1
        seq = self._seq
        self._start_pump(seq)
        start = time.time()
        st = dict.fromkeys(["first", "wait_bucket", "wait_inbound", "consume", "wait_forward", "free"], 0.0)
        self._recv_start = start
        acc = {"buckets": 0, "bytes": 0}
        self._acc = dict.fromkeys(["ready_wait", "credit_wait", "write"], 0.0)
        handoff: asyncio.Queue = asyncio.Queue()
        assembling: dict[str, list] = {}  # export="shard": name -> [full tensor bytes, bytes still missing]
        self._recv_model_type = None

        async def run_chain(c):
            pieces = self._receive_chain(c, seq, global_steps, st, acc)
            if self.export == "shard":
                tensors = self._assemble(pieces, assembling)
            else:
                tensors = merge_weight_chunks(pieces, self.chain_bucket)
            try:
                async for name, weight in tensors:
                    taken = asyncio.Event()
                    await handoff.put(("item", name, weight, taken))
                    await taken.wait()  # the consumer asked for the next tensor: it is done with this one
                await handoff.put(("done", c, None, None))
            except asyncio.CancelledError:
                raise
            except BaseException as e:
                await handoff.put(("error", c, e, None))

        tasks = [asyncio.ensure_future(run_chain(c)) for c in self._roles]
        remaining = len(tasks)
        try:
            while remaining:
                kind, a, b, taken = await handoff.get()
                if kind == "item":
                    t0 = time.perf_counter()
                    for name, weight in self._unfuse(a, b):
                        yield name, weight
                    st["consume"] += time.perf_counter() - t0
                    taken.set()
                elif kind == "done":
                    remaining -= 1
                else:
                    raise b
        except BaseException as e:  # incl. GeneratorExit: the consumer (engine load) stopped mid-sync
            await self._notify_abort(seq, e)
            raise
        finally:
            await self._reap(tasks)
            await self._stop_pump()
        dt = time.time() - start
        if self.verify:
            self._vlog("summary", buckets=acc["buckets"], ok_first_check=getattr(self, "_v_ok", 0), secs=round(dt, 3))
            self._v_ok = 0
        if self._stats_on:
            forwards = any(r.nexts for r in self._roles.values())
            fwd = {k: self._acc[k] for k in ("credit_wait", "write")} if forwards else {}
            self._stats("recv", dt, acc["buckets"], acc["bytes"], {**st, **fwd})
        logger.info(
            f"mori rank {self.rank} received {acc['buckets']} buckets from {len(self._roles)} chains, "
            f"{acc['bytes'] / 2**30:.2f} GiB in {dt:.2f}s"
        )

    @staticmethod
    async def _assemble(pieces, assembling: dict) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """export="shard": copy each received piece to its place in the full tensor (shared by all stripes, which
        bring the other ranks' pieces) and yield a tensor once every byte of it arrived."""
        async for meta, view in pieces:
            entry = assembling.get(meta.name)
            if entry is None:
                total = meta.shape.numel() * meta.dtype.itemsize
                entry = assembling[meta.name] = [torch.empty(total, dtype=torch.uint8, device=view.device), total]
            entry[0][meta.chunk_offset : meta.chunk_offset + meta.chunk_size].copy_(view, non_blocking=True)
            entry[1] -= meta.chunk_size
            if entry[1] == 0:
                del assembling[meta.name]
                yield meta.name, entry[0].view(meta.dtype).view(meta.shape)

    def _unfuse(self, name: str, tensor: torch.Tensor):
        """export="shard" ships the trainer's own parameters, e.g. Transformers 5 packed MoE experts; split them into
        the per-expert checkpoint names, as the FSDP engine's full export does (views, no copies)."""
        if self.export != "shard" or self._recv_model_type is None:
            yield name, tensor
            return
        yield from self._unfuse_moe_params([(name, tensor)], self._recv_model_type)

    async def _receive_chain(
        self, c, seq, global_steps, st, acc
    ) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
        ro = self._roles[c]
        k, early = 0, {}
        # per slot: the task that frees it once its forward writes have left and the consumer is done with it; the
        # loop moves on to the next bucket meanwhile, so up to num_slots buckets per stripe are in flight here
        release: list[asyncio.Future | None] = [None] * self.num_slots
        forward = None
        try:
            while True:
                t0 = time.perf_counter()
                while k not in early:
                    msg = await self._timed(self._buckets[c].get(), f"bucket {k} of chain {c} from {ro.prev.key}")
                    self._check_abort()
                    early[msg["k"]] = msg
                msg = early.pop(k)
                st["wait_bucket"] += time.perf_counter() - t0
                if not st["first"]:  # until the first bucket of any chain was announced
                    st["first"] = max(time.time() - self._recv_start, 1e-9)
                steps = msg["global_steps"]
                if global_steps is not None and steps is not None and steps != global_steps:
                    raise RuntimeError(f"mori: got weights of step {steps} while expecting step {global_steps}")
                if msg.get("model_type") is not None:
                    self._recv_model_type = msg["model_type"]
                j = msg["slot"]
                assert j == k % self.num_slots, f"bucket {k} of chain {c} arrived in slot {j}"
                for r in release:  # surface a failed forward or free early
                    if r is not None and r.done():
                        r.result()
                if release[j] is not None:  # already done: prev wrote bucket k into slot j only after it got the free
                    await release[j]
                    release[j] = None
                t0 = time.perf_counter()
                if msg["uid"] is not None and msg.get("notified", True):  # RDMA hop; XGMI hops raise no notification
                    await self._wait_inbound(ro.prev.key, msg["uid"], f"arrival of bucket {k} of chain {c}")
                st["wait_inbound"] += time.perf_counter() - t0
                if self._v_recv and msg.get("sums") is not None:
                    await self._verify_arrival(c, k, j, msg, ro.prev.key)
                forward = None
                if ro.nexts:  # next node's holder and/or the other GPUs of my node
                    forward = asyncio.ensure_future(
                        self._write_next(
                            seq,
                            c,
                            k,
                            j,
                            msg["metas"],
                            msg["used"],
                            msg["is_last"],
                            steps,
                            msg.get("sums"),
                            model_type=msg.get("model_type"),
                        )
                    )
                buf = self._slot(c, j)
                for tensor_meta in msg["metas"]:
                    yield tensor_meta, buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
                # the consumer asked for the next tensor, so its reads of this bucket's views are queued on the
                # current stream: free the slot once they ran and the forward writes have left
                release[j] = asyncio.ensure_future(self._release(seq, c, j, forward, self._mark_ready(), st))
                forward = None
                acc["buckets"] += 1
                acc["bytes"] += msg["used"]
                if msg["is_last"]:
                    break
                k += 1
            for j, r in enumerate(release):
                if r is not None:
                    await r
                    release[j] = None
            if ro.nexts:
                await self._drain_credits(c, "receive_weights")
        finally:
            await self._reap([forward, *release])
