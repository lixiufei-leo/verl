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
"""CPU tests for the mori checkpoint engine with an in-process fake of MoRI-IO.

No GPU, NIC or Ray: every engine runs in one asyncio loop and talks over real ZMQ on localhost. The fake IOEngine
implements WRITE as a byte copy that lands after a random delay, followed by a target-side notification, so buckets
of different chains complete out of order. The test drives engines the way CheckpointEngineManager does:
prepare -> build_topology -> init_process_group -> send/receive -> finalize, several syncs in a row.
"""

import asyncio
import importlib
import importlib.machinery
import itertools
import pickle
import random
import sys
import threading
import time
import types

import pytest
import torch

BUCKET = 64 << 10  # small, so models span many buckets and chains
_ENGINES = []  # every engine a test creates; closed by the fixture


def _fake_mori(early_notify: bool = False):
    """Fake `mori` / `mori.io` modules backed by a dict of registered tensors. With early_notify the target-side
    notification is visible before the data lands (models a NIC whose completion overtakes the payload)."""
    fab = {
        "engines": {},
        "mems": {},
        "ids": itertools.count(1),
        "lock": threading.Lock(),
        "writes": [],
        "threads": [],
        "node_of": {},
    }  # engine key -> node, filled by the test after prepare(); same node + XGMI on both = XGMI

    class Status:
        def __init__(self, done_at, ok=True):
            self.done_at, self.ok = done_at, ok

        def InProgress(self):
            return time.monotonic() < self.done_at

        def Succeeded(self):
            return not self.InProgress() and self.ok

        def Failed(self):
            return not self.InProgress() and not self.ok

        def Message(self):
            return ""

        def WaitFor(self, ms):
            time.sleep(max(0.0, min(self.done_at - time.monotonic(), ms / 1000)))

    class EngineDesc:
        def __init__(self, key):
            self.key = key

        def pack(self):
            return pickle.dumps(self.key)

        @staticmethod
        def unpack(b):
            return EngineDesc(pickle.loads(b))

    class MemoryDesc:
        def __init__(self, mid, size):
            self.id, self.size = mid, size

        def pack(self):
            return pickle.dumps((self.id, self.size))

        @staticmethod
        def unpack(b):
            return MemoryDesc(*pickle.loads(b))

    class IOEngine:
        def __init__(self, key, config):
            self.key, self.remotes, self.inbound, self.uid = key, set(), {}, itertools.count(1)
            self.xgmi = False
            fab["engines"][key] = self

        def create_backend(self, backend_type, config):
            if backend_type == "XGMI":
                self.xgmi = True

        def get_engine_desc(self):
            return EngineDesc(self.key)

        def register_remote_engine(self, desc):
            self.remotes.add(desc.key)

        def register_torch_tensor(self, t):
            mid = next(fab["ids"])
            fab["mems"][mid] = (self.key, t)
            return MemoryDesc(mid, t.numel())

        def allocate_transfer_uid(self):
            return next(self.uid)

        def write(self, local, local_off, remote, remote_off, size, uid):
            owner, dst = fab["mems"][remote.id]
            assert owner in self.remotes, "write to an engine that was not registered"
            assert local_off + size <= local.size and remote_off + size <= remote.size, "out of bounds"
            data = fab["mems"][local.id][1][local_off : local_off + size].clone()  # the NIC reads the source now
            target = fab["engines"][owner]
            node = fab["node_of"].get(self.key)
            if self.xgmi and target.xgmi and node is not None and node == fab["node_of"].get(owner):
                # like MoRI's XGMI backend: the copy is done when the initiator's status completes, and the target
                # gets no inbound notification
                dst[remote_off : remote_off + size].copy_(data)
                fab["writes"].append((self.key, owner, remote_off))
                fab["xgmi_writes"] = fab.get("xgmi_writes", 0) + 1
                return Status(0)
            delay = random.uniform(0, 0.003)

            def land():
                time.sleep(delay + (0.02 if early_notify else 0))
                dst[remote_off : remote_off + size].copy_(data)
                if not early_notify:
                    with fab["lock"]:
                        target.inbound[(self.key, uid)] = Status(0)

            if early_notify:
                with fab["lock"]:
                    target.inbound[(self.key, uid)] = Status(0)
            th = threading.Thread(target=land, daemon=True)
            th.start()
            fab["threads"].append(th)
            fab["writes"].append((self.key, owner, remote_off))
            return Status(time.monotonic() + (0 if early_notify else delay + 0.001))

        def pop_inbound_transfer_status(self, remote_key, uid):
            with fab["lock"]:
                return self.inbound.pop((remote_key, uid), None)

        def pop_peer_failure(self):
            return None

        def deregister_memory(self, mem):
            fab["mems"].pop(mem.id, None)

    io = types.ModuleType("mori.io")
    io.__spec__ = importlib.machinery.ModuleSpec("mori.io", None)
    for name, obj in dict(
        IOEngine=IOEngine,
        EngineDesc=EngineDesc,
        MemoryDesc=MemoryDesc,
        IOEngineConfig=lambda host="", port=0: None,
        RdmaBackendConfig=lambda **kw: None,
        XgmiBackendConfig=lambda **kw: None,
        BackendType=types.SimpleNamespace(RDMA="RDMA", XGMI="XGMI"),
    ).items():
        setattr(io, name, obj)
    mori = types.ModuleType("mori")
    mori.__spec__ = importlib.machinery.ModuleSpec("mori", None)
    mori.io = io
    return mori, io, fab


@pytest.fixture
def mori_env(monkeypatch):
    """Install the fake MoRI (its module is imported lazily by the engine) and return (engine module, factory)."""
    fabs = []

    def install(early_notify=False):
        mori, io, fab = _fake_mori(early_notify)
        monkeypatch.setitem(sys.modules, "mori", mori)
        monkeypatch.setitem(sys.modules, "mori.io", io)
        fabs.append(fab)
        return fab

    install()
    mod = importlib.import_module("verl.checkpoint_engine.mori_checkpoint_engine")
    yield mod, install, fabs
    for fab in fabs:  # let in-flight fake writes land before the next test
        for th in fab["threads"]:
            th.join()
    while _ENGINES:
        _ENGINES.pop().close()


def _make_weights(seed):
    g, r = torch.Generator().manual_seed(seed), random.Random(seed)
    ws = [("embed", torch.randn(300, 77, generator=g).to(torch.bfloat16))]
    dtypes = [torch.bfloat16, torch.float32, torch.int64, torch.float16]
    for i in range(30):
        n, dt = r.randint(1, 3000), dtypes[i % 4]
        t = torch.randint(-1000, 1000, (n,), generator=g) if dt == torch.int64 else torch.randn(n, generator=g).to(dt)
        ws.append((f"layer{i}.w", t))
    # low-precision rollout: quantized on the trainer side and exported as (weight, scale) pairs; the engine moves
    # bytes, so every dtype must arrive bit-exact
    ws.append(("mlp.fp8.weight", torch.randn(120, 70, generator=g).to(torch.float8_e4m3fn)))
    ws.append(("mlp.fp8.weight_scale_inv", torch.rand(1, 3, generator=g) + 0.5))
    ws.append(("attn.fp8.weight", torch.randn(333, generator=g).to(torch.float8_e5m2)))
    ws.append(("mlp.mxfp4.weight_packed", torch.randint(0, 256, (257, 32), dtype=torch.uint8, generator=g)))
    if hasattr(torch, "float8_e8m0fnu"):
        ws.append(("mlp.mxfp4.weight_scale", (torch.rand(257, generator=g) * 8 + 0.5).to(torch.float8_e8m0fnu)))
    ws.append(("big", torch.randn(100_000, generator=g)))  # 400 KB: spans many buckets, merged on receive
    ws.append(("tail.bias", torch.randn(3, generator=g).to(torch.bfloat16)))
    return ws


def _same(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
    )


def _new(mod, num_chains, node="n0", **kw):
    e = mod.MoriCheckpointEngine(
        bucket_size=BUCKET, num_chains=num_chains, host="127.0.0.1", device="cpu", timeout_s=30, node_id=node, **kw
    )
    _ENGINES.append(e)
    return e


def _wire(actors, rollouts, fab=None):
    meta = [e.prepare() for e in actors + rollouts]
    if fab is not None:
        fab["node_of"].update({e._key: e.node_id for e in actors + rollouts})
    akw, rkw = type(actors[0]).build_topology(len(actors), len(rollouts), meta)
    for i, e in enumerate(actors):
        e.init_process_group(**{k: v[i] for k, v in akw.items()})
    for i, e in enumerate(rollouts):
        e.init_process_group(**{k: v[i] for k, v in rkw.items()})
    return meta


async def _consume(e, step, delay=0.0, fail_at=None):
    got = {}
    async for name, w in e.receive_weights(global_steps=step):
        assert name not in got, f"{name} received twice"
        got[name] = w.clone()
        if fail_at is not None and len(got) == fail_at:
            raise ValueError("simulated load_weights failure")
        if delay:
            await asyncio.sleep(delay)
    return got


def _check_traffic(fab, actors, rollouts, nodes, P):
    """v2 traffic: each rollout node takes each stripe exactly once from outside, and one GPU of the node (the
    stripe's holder) copies it to the others; different stripes use different holders (so different NICs)."""
    node_of = {e._key: e.node_id for e in actors + rollouts}
    cb, slots = actors[0].chain_bucket, actors[0].num_slots
    inbound, local = {}, {}  # (node, stripe) -> writers from outside / inside the node
    for src, dst, off in fab["writes"]:
        key = (node_of[dst], off // cb // slots)
        (inbound if node_of[src] != node_of[dst] else local).setdefault(key, set()).add(src)
    for ni, size in enumerate(nodes):
        for c in range(P):
            assert len(inbound.get((f"r{ni}", c), ())) == 1, (ni, c, inbound.get((f"r{ni}", c)))
            if size > 1:
                assert len(local.get((f"r{ni}", c), ())) == 1, (ni, c, local.get((f"r{ni}", c)))
        if size > 1:
            holders = {w for c in range(P) for w in local[(f"r{ni}", c)]}
            assert len(holders) == min(P, size)
    if max(nodes) > 1:  # same-node copies took the XGMI path, which raises no inbound notification
        assert fab.get("xgmi_writes", 0) > 0
    fab["xgmi_writes"] = 0


def test_registered(mori_env):
    mod, _, _ = mori_env
    from verl.checkpoint_engine.base import CheckpointEngineRegistry

    assert CheckpointEngineRegistry.get("mori") is mod.MoriCheckpointEngine


@pytest.mark.parametrize(
    "n_actors, nodes, num_chains, num_slots",
    [
        (2, [3], 1, 2),
        (4, [3], 8, 2),
        (4, [3], 2, 2),
        (4, [2], 8, 2),
        (2, [6], 8, 2),
        (4, [3, 3], 8, 2),
        (8, [4, 4], 8, 2),
        (2, [2, 2, 2, 2], 8, 2),
        (4, [3, 3], 1, 2),
        (8, [2, 2], 8, 2),
        (2, [3], 1, 4),
        (4, [3, 3], 8, 3),
    ],
    ids=[
        "single-stripe",
        "stripe-per-actor",
        "fewer-stripes-than-actors",
        "more-stripes-than-gpus",
        "one-node-2x6",
        "2-nodes",
        "2-nodes-8-stripes",
        "4-nodes",
        "2-nodes-1-stripe",
        "stripes-wrap-gpus",
        "single-stripe-4-slots",
        "2-nodes-3-slots",
    ],
)
def test_every_rollout_gets_every_tensor(mori_env, n_actors, nodes, num_chains, num_slots):
    """Trainer on its own node, rollout workers on len(nodes) nodes. Every tensor must arrive bit-exact, and traffic
    must follow v2: each rollout node takes each stripe exactly once from outside, and one GPU of the node (the
    stripe's holder) copies it to the others; different stripes use different holders (so different NICs)."""
    mod, _, fabs = mori_env
    fab = fabs[0]
    actors = [_new(mod, num_chains, node="trainer", is_master=(i == 0), num_slots=num_slots) for i in range(n_actors)]
    rollouts = [
        _new(mod, num_chains, node=f"r{ni}", num_slots=num_slots) for ni, size in enumerate(nodes) for _ in range(size)
    ]

    async def run():
        first = None
        for step in range(3):  # buckets, registrations and sockets must be reused across syncs
            meta = _wire(actors, rollouts, fab)
            first = first or meta
            assert meta == first
            ws = _make_weights(step)
            fab["writes"].clear()
            res = await asyncio.gather(
                *[a.send_weights(iter(ws), global_steps=step) for a in actors],
                *[_consume(r, step, 0.002 if i == 1 else 0) for i, r in enumerate(rollouts)],
            )
            for got in res[n_actors:]:
                assert set(got) == {n for n, _ in ws}
                for n, w in ws:
                    assert _same(got[n], w), n
            P = actors[0]._P
            assert P == min(num_chains, n_actors)
            _check_traffic(fab, actors, rollouts, nodes, P)
            for e in actors + rollouts:
                e.finalize()

    asyncio.run(run())


class _ShardedTrainerRank:
    """One rank of a trainer whose parameters are cut along dim 0 into `world` shards (torch.chunk, as FSDP2
    Shard(0)); exposes get_per_tensor_param_shard() like verl's training engines: (name, flat local shard, spec)."""

    def __init__(self, rank, world, weights, replicated=(), model_type=None):
        self.rank, self.world, self.weights, self.replicated = rank, world, weights, set(replicated)
        self.model_config = types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type=model_type))

    def get_per_tensor_param_shard(self):
        def gen():
            for name, w in self.weights:
                if name in self.replicated or w.dim() == 0:  # whole tensor on every rank: rank 0 sends it
                    yield (
                        name,
                        w.reshape(-1),
                        types.SimpleNamespace(full_shape=tuple(w.shape), place=0, contributes=self.rank == 0),
                    )
                    continue
                parts = list(torch.chunk(w, self.world, dim=0))  # can be fewer than world: the last ranks get none
                local = parts[self.rank] if self.rank < len(parts) else w[:0]
                first_row = sum(p.shape[0] for p in parts[: self.rank])
                row = w[0].numel() if w.shape[0] else 0
                yield (
                    name,
                    local.reshape(-1),
                    types.SimpleNamespace(full_shape=tuple(w.shape), place=first_row * row, contributes=True),
                )

        return gen(), None


def _moe_weights(seed):
    """_make_weights plus Transformers-5 packed MoE experts, an empty and a 0-dim tensor."""
    g = torch.Generator().manual_seed(seed + 100)
    ws = _make_weights(seed)
    ws.append(("model.layers.0.mlp.experts.gate_up_proj", torch.randn(5, 6, 8, generator=g).to(torch.bfloat16)))
    ws.append(("model.layers.0.mlp.experts.down_proj", torch.randn(5, 8, 3, generator=g).to(torch.bfloat16)))
    ws.append(("model.layers.0.mlp.gate.weight", torch.randn(5, 8, generator=g)))
    ws.append(("empty.w", torch.empty(0, 5)))
    ws.append(("scalar", torch.tensor(1.5)))
    return ws


def _unfused(ws):
    """What the rollout must receive: packed experts under their per-expert checkpoint names."""
    out = []
    for name, w in ws:
        if name.endswith(".mlp.experts.gate_up_proj"):
            gate, up = w.chunk(2, dim=1)
            base = name.removesuffix(".gate_up_proj")
            for e in range(w.shape[0]):
                out += [(f"{base}.{e}.gate_proj.weight", gate[e]), (f"{base}.{e}.up_proj.weight", up[e])]
        elif name.endswith(".mlp.experts.down_proj"):
            base = name.removesuffix(".down_proj")
            out += [(f"{base}.{e}.down_proj.weight", w[e]) for e in range(w.shape[0])]
        else:
            out.append((name, w))
    return out


@pytest.mark.parametrize(
    "n_actors, nodes, num_slots",
    [(2, [3], 2), (4, [3, 3], 2), (8, [4, 4], 2), (3, [2, 2], 3), (8, [2, 2, 2], 2)],
    ids=["1-node", "2-nodes", "2-nodes-8-ranks", "uneven-shards-3-slots", "3-nodes-stripes-wrap-gpus"],
)
def test_shard_export_assembles_full_tensors(mori_env, n_actors, nodes, num_slots):
    """export="shard": every trainer rank sends only its own dim-0 shards (stripe = rank); every rollout worker must
    assemble every full tensor bit-exact, with packed MoE experts split into per-expert names."""
    mod, _, fabs = mori_env
    fab = fabs[0]
    kw = dict(export="shard", num_slots=num_slots)
    actors = [_new(mod, 8, node="trainer", is_master=(i == 0), **kw) for i in range(n_actors)]
    rollouts = [_new(mod, 8, node=f"r{ni}", **kw) for ni, size in enumerate(nodes) for _ in range(size)]
    assert all(e.consumes_training_engine for e in actors)

    async def run():
        for step in range(2):
            _wire(actors, rollouts, fab)
            ws = _moe_weights(step)
            ranks = [_ShardedTrainerRank(i, n_actors, ws, ["embed"], "qwen3_moe") for i in range(n_actors)]
            fab["writes"].clear()
            res = await asyncio.gather(
                *[a.send_weights(r, global_steps=step) for a, r in zip(actors, ranks, strict=True)],
                *[_consume(r, step, 0.002 if i == 1 else 0) for i, r in enumerate(rollouts)],
            )
            want = _unfused(ws)
            for got in res[n_actors:]:
                assert set(got) == {n for n, _ in want}
                for n, w in want:
                    assert _same(got[n], w), n
            assert actors[0]._P == n_actors
            _check_traffic(fab, actors, rollouts, nodes, n_actors)
            for e in actors + rollouts:
                e.finalize()

    asyncio.run(run())


def test_shard_export_needs_a_stripe_per_rank(mori_env):
    mod, _, _ = mori_env
    actors = [_new(mod, 2, node="trainer", export="shard") for _ in range(4)]
    rollouts = [_new(mod, 2, node="r0", export="shard") for _ in range(2)]
    with pytest.raises(AssertionError, match="num_chains >= trainer world size"):
        _wire(actors, rollouts)


def test_version_mismatch_fails_everyone_fast(mori_env):
    mod, _, _ = mori_env
    actors = [_new(mod, 4, node="trainer", is_master=(i == 0)) for i in range(4)]
    rollouts = [_new(mod, 4, node=f"r{i % 2}") for i in range(3)]
    _wire(actors, rollouts)

    async def run():
        return await asyncio.gather(
            *[a.send_weights(iter(_make_weights(7)), global_steps=7) for a in actors],
            *[_consume(r, 8) for r in rollouts],
            return_exceptions=True,
        )

    t = time.time()
    res = asyncio.run(run())
    assert time.time() - t < 5  # far below timeout_s: the failure is propagated, not waited out
    assert all(isinstance(x, RuntimeError) for x in res)
    assert any("step 7 while expecting step 8" in str(x) for x in res)


def test_consumer_failure_fails_everyone_fast(mori_env):
    mod, _, _ = mori_env
    actors = [_new(mod, 4, node="trainer", is_master=(i == 0)) for i in range(4)]
    rollouts = [_new(mod, 4, node=f"r{i % 2}") for i in range(3)]
    _wire(actors, rollouts)

    async def run():
        return await asyncio.gather(
            *[a.send_weights(iter(_make_weights(3)), global_steps=3) for a in actors],
            _consume(rollouts[0], 3),
            _consume(rollouts[1], 3, fail_at=10),
            _consume(rollouts[2], 3),
            return_exceptions=True,
        )

    t = time.time()
    res = asyncio.run(run())
    assert time.time() - t < 5
    assert isinstance(res[5], ValueError)
    assert all(isinstance(x, RuntimeError) for i, x in enumerate(res) if i != 5)


def test_verify_recv_waits_for_late_data(mori_env):
    """verify="recv": a receiver that sees the notification before the bytes waits for them instead of using them."""
    mod, install, _ = mori_env
    install(early_notify=True)
    actors = [_new(mod, 2, node="trainer", is_master=(i == 0), verify="recv") for i in range(2)]
    rollouts = [_new(mod, 2, node=f"r{i % 2}", verify="recv") for i in range(3)]
    _wire(actors, rollouts)
    ws = _make_weights(5)

    async def run():
        return await asyncio.gather(
            *[a.send_weights(iter(ws), global_steps=5) for a in actors], *[_consume(r, 5) for r in rollouts]
        )

    res = asyncio.run(run())
    for got in res[2:]:
        for n, w in ws:
            assert _same(got[n], w), n
