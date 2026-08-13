"""Edge-aware graph controller for NeuroFold v8 (NumPy, CPU).

Architecture:
    h_i        = tanh(W_node  node_i + b)
    e_ij       = tanh(W_edge  edge_ij + b)
    m_ij       = tanh(W_msg [h_i, h_j, e_ij] + b)
    a_ij       = softmax_j( w_att [h_i, h_j, e_ij] )
    h_i'       = tanh(W_upd [h_i, sum_j a_ij m_ij] + b)        x n_layers
    logits_i   = w_logit h_i''            -> which bead to act on
    (alpha,delta)_i = tanh(W_cont h_i'')  -> how to act on it
    value      = w_val mean_i(h_i'')      -> critic for RL

The message pathway is the object under test.  `local_only=True` zeroes the
aggregated message (and freezes those parameters), giving a controller that can
still see every per-node feature but cannot condition on WHICH partner a bead
is in contact with.  Comparing the two is acceptance test A1; in v7 the
equivalent comparison was statistically indistinguishable on all five tasks.

A history encoder (optionally recurrent) feeds the partially observable
environment state into every node, for A2.
"""
from __future__ import annotations

import numpy as np


def _softmax_seg(scores, index, n):
    """Segment-wise softmax of `scores` grouped by destination node `index`."""
    out = np.zeros_like(scores)
    mx = np.full(n, -np.inf)
    np.maximum.at(mx, index, scores)
    e = np.exp(scores - mx[index])
    s = np.zeros(n)
    np.add.at(s, index, e)
    out = e / (s[index] + 1e-12)
    return out


class GraphPolicySpec:
    """Shapes and flat-vector packing for the controller."""

    def __init__(self, node_dim, edge_dim, hidden=16, msg=24, layers=3,
                 hist_dim=40, hist_hidden=8, recurrent=False):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.H = hidden
        self.M = msg
        self.L = layers
        self.hist_dim = hist_dim
        self.HH = hist_hidden
        self.recurrent = recurrent
        H, M, HH = hidden, msg, hist_hidden
        spec = [("W_node", (H, node_dim)), ("b_node", (H,)),
                ("W_hist", (HH, hist_dim)), ("b_hist", (HH,)),
                ("W_hin", (H, HH)),
                ("W_edge", (H, edge_dim)), ("b_edge", (H,))]
        for l in range(layers):
            spec += [(f"W_msg{l}", (M, 3 * H)), (f"b_msg{l}", (M,)),
                     (f"w_att{l}", (3 * H,)), (f"b_att{l}", ()),
                     (f"W_upd{l}", (H, H + M)), (f"b_upd{l}", (H,))]
        if recurrent:
            spec += [("W_gru_z", (HH, HH + HH)), ("W_gru_r", (HH, HH + HH)),
                     ("W_gru_h", (HH, HH + HH))]
        # Edge-selection head: the action is a PAIR, so the controller scores
        # edges, not nodes.  The full model scores from [h_i, h_j, e_ij]; the
        # local-only control sees h_i and h_j but no edge embedding and no
        # message passing, so it can still choose a pair but cannot use the
        # relational context that says WHICH pair is pathological.
        spec += [("w_edge_sel", (3 * H,)), ("b_edge_sel", ()),
                 ("w_str", (3 * H,)), ("b_str", ()),
                 ("w_val", (H,)), ("b_val", ())]
        self.spec = spec
        self.slices = {}
        i = 0
        for k, s in spec:
            n = int(np.prod(s)) if s else 1
            self.slices[k] = (i, i + n, s)
            i += n
        self.dim = i
        a, b, _ = self.slices["W_edge"]
        self.edge_param_index = np.arange(self.slices["W_edge"][0],
                                          self.slices[f"b_upd{layers-1}"][1])
        # indices belonging strictly to the message pathway
        msg_keys = [f"W_msg{l}" for l in range(layers)] + \
                   [f"b_msg{l}" for l in range(layers)] + \
                   [f"w_att{l}" for l in range(layers)] + \
                   [f"b_att{l}" for l in range(layers)] + \
                   ["W_edge", "b_edge"]
        idx = []
        for k in msg_keys:
            a, b, _ = self.slices[k]
            idx.extend(range(a, b))
        self.message_param_index = np.asarray(sorted(idx), dtype=int)

    def unpack(self, vec):
        v = np.asarray(vec, float)
        return {k: (float(v[a]) if s == () else v[a:b].reshape(s))
                for k, (a, b, s) in self.slices.items()}

    def init(self, rng, scale=0.25):
        v = rng.normal(0, scale, self.dim) / np.sqrt(self.H)
        return v


class GraphPolicy:
    def __init__(self, spec: GraphPolicySpec, local_only=False):
        self.spec = spec
        self.local_only = local_only

    def encode(self, p, obs, hstate=None):
        s = self.spec
        node = obs["node"]
        n = node.shape[0]
        h = np.tanh(node @ p["W_node"].T + p["b_node"])
        hist = np.tanh(obs["history"].ravel() @ p["W_hist"].T + p["b_hist"])
        if s.recurrent and hstate is not None:
            cat = np.concatenate([hstate, hist])
            z = 1 / (1 + np.exp(-(p["W_gru_z"] @ cat)))
            r = 1 / (1 + np.exp(-(p["W_gru_r"] @ cat)))
            hh = np.tanh(p["W_gru_h"] @ np.concatenate([r * hstate, hist]))
            hist = (1 - z) * hstate + z * hh
        h = h + (hist @ p["W_hin"].T)[None, :]

        if not self.local_only:
            ei, ej = obs["edge_index"]
            e = np.tanh(obs["edge"] @ p["W_edge"].T + p["b_edge"])
            for l in range(s.L):
                cat = np.concatenate([h[ei], h[ej], e], axis=1)
                m = np.tanh(cat @ p[f"W_msg{l}"].T + p[f"b_msg{l}"])
                att = cat @ p[f"w_att{l}"] + p[f"b_att{l}"]
                a = _softmax_seg(att, ei, n)
                agg = np.zeros((n, s.M))
                np.add.at(agg, ei, a[:, None] * m)
                h = np.tanh(np.concatenate([h, agg], axis=1) @ p[f"W_upd{l}"].T
                            + p[f"b_upd{l}"])
        else:
            zero = np.zeros((n, s.M))
            for l in range(s.L):
                h = np.tanh(np.concatenate([h, zero], axis=1) @ p[f"W_upd{l}"].T
                            + p[f"b_upd{l}"])
        return h, hist

    def forward(self, vec, obs, hstate=None):
        p = self.spec.unpack(vec)
        h, hist = self.encode(p, obs, hstate)
        ei, ej = obs["edge_index"]
        if self.local_only:
            e = np.zeros((len(ei), self.spec.H))
        else:
            e = np.tanh(obs["edge"] @ p["W_edge"].T + p["b_edge"])
        cat = np.concatenate([h[ei], h[ej], e], axis=1)
        logits = cat @ p["w_edge_sel"] + p["b_edge_sel"]
        strength = 0.5 * (1.0 + np.tanh(cat @ p["w_str"] + p["b_str"]))
        value = float(h.mean(0) @ p["w_val"] + p["b_val"])
        return logits, strength, value, hist, (ei, ej)

    def act(self, vec, obs, hstate=None):
        logits, strength, _, hist, (ei, ej) = self.forward(vec, obs, hstate)
        if len(ei) == 0:
            # No contact is above the edge threshold: there is nothing to modulate.
            # argmax on an empty array raises, which would abort a graded run.
            return (0, 0, 0.0), hist
        k = int(np.argmax(logits))
        return (int(ei[k]), int(ej[k]), float(strength[k])), hist

    def sample(self, vec, obs, rng, temperature=1.0, sigma=0.15, hstate=None):
        logits, strength, value, hist, (ei, ej) = self.forward(vec, obs, hstate)
        z = logits / max(temperature, 1e-6)
        z -= z.max()
        pr = np.exp(z)
        pr /= pr.sum()
        k = int(rng.choice(len(pr), p=pr))
        st = float(np.clip(rng.normal(strength[k], sigma), 0.0, 1.0))
        logp = (np.log(pr[k] + 1e-12)
                - 0.5 * ((st - strength[k]) ** 2) / sigma ** 2)
        return (int(ei[k]), int(ej[k]), st), float(logp), value, hist


def make_spec(env, **kw):
    obs = env.observe()
    return GraphPolicySpec(node_dim=obs["node"].shape[1],
                           edge_dim=obs["edge"].shape[1],
                           hist_dim=obs["history"].size, **kw)
