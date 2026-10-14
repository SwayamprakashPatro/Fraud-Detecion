"""Model service: runs the repo's own code (src/) and exposes scores, metrics and facts.

Nothing here is typed in. Everything is read from, or computed with:
  graph.gpickle, Cleaned_Database.csv           via src.data_loader.FraudGraphDataset
  outputs/models/best_graphsage.pt              via src.models.graphsage.FraudGraphSAGE
  outputs/logs/*.json, final_report.md          the files the training run wrote
"""
import datetime
import json
import os
import pickle
import re
import threading

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import config
from src.data_loader import FraudGraphDataset
from src.models.graphsage import FraudGraphSAGE

PREFIX = {"account": "A", "device": "D", "merchant": "M", "upi": "U"}


def _node_id(name: str) -> str:
    kind, num = name.split("_", 1)
    return PREFIX[kind] + num


def _confusion(y, prob, th):
    pr = (prob >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.astype(int), pr, labels=[0, 1]).ravel()
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), precision=p, recall=r,
                f1=2 * p * r / max(1e-12, p + r), accuracy=(tp + tn) / max(1, len(y)))


class ModelService:
    def __init__(self):
        self.lock = threading.RLock()
        self.loaded = False
        self._payload = None

    # ---------------------------------------------------------------- loading
    def load(self):
        with self.lock:
            ds = FraudGraphDataset()
            data, meta = ds.load_data()
            ck = torch.load(config.BEST_MODEL_PATH, map_location="cpu")
            sd = ck["model_state_dict"]
            hidden = sd["conv1.lin_l.bias"].shape[0]
            model = FraudGraphSAGE(node_in_channels=meta["node_feature_dim"], edge_in_channels=meta["num_features"],
                                   hidden_channels=hidden, out_channels=hidden, num_layers=config.NUM_LAYERS, dropout=0.0)
            model.load_state_dict(sd)
            model.eval()
            probs = model.predict_proba(data.x, data.edge_index, data.tx_src, data.tx_dst, data.tx_edge_attr).numpy()

            with open(config.GRAPH_PATH, "rb") as f:
                G = pickle.load(f)
            csv_ids = pd.read_csv(config.CSV_DATA_PATH, usecols=["Transaction_ID"])["Transaction_ID"].astype(int)
            tx, dev, mer, upi = {}, {}, {}, {}
            for u, v, k, d in G.edges(keys=True, data=True):
                t = d.get("type")
                if t == "transaction":
                    tx[int(k)] = (u, v, d)
                elif t == "used_device":
                    dev[int(k)] = v
                elif t == "merchant":
                    mer[int(k)] = v
                elif t == "upi_used":
                    upi[int(k)] = v
            ids = [int(i) for i in sorted(csv_ids) if int(i) in tx]  # same order the loader uses
            if len(ids) != len(probs):
                raise RuntimeError(f"Row mismatch: {len(ids)} payments vs {len(probs)} scores")

            self.ds, self.data, self.meta, self.ck, self.model, self.G = ds, data, meta, ck, model, G
            self.probs, self.ids, self.tx, self.dev, self.mer, self.upi = probs, ids, tx, dev, mer, upi
            self.index = {tid: i for i, tid in enumerate(ids)}
            self.threshold = float(ck.get("best_threshold", 0.5))
            self.split = np.zeros(len(ids), int)
            self.split[data.val_mask.numpy()] = 1
            self.split[data.test_mask.numpy()] = 2
            self.y = data.y.numpy()
            self.col_means = data.tx_edge_attr.mean(0)
            self.report = self._json(config.EVAL_REPORT_PATH)
            self.history = self._json(config.HISTORY_PATH)
            self.claims = self._claims()
            self._reality = None
            self._payload = None
            self.loaded = True
            self.payload()  # build once now, including the stress tests, so requests are instant

    @staticmethod
    def _json(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except OSError:
            return None

    def _claims(self):
        out = {}
        try:
            with open(os.path.join(config.BASE_DIR, "final_report.md"), encoding="utf-8") as f:
                md = f.read()
        except OSError:
            return out
        g = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, md))
        out = dict(f1=g(r"F1-Score\*\*\s*\|\s*\*\*([\d.]+)\*\*"), auc=g(r"ROC-AUC\*\*\s*\|\s*\*\*([\d.]+)\*\*"),
                   pr=g(r"PR-AUC\*\*\s*\|\s*\*\*([\d.]+)\*\*"))
        m = re.search(r"checkpoint from \*\*Epoch (\d+)\*\*", md)
        out["epoch"] = int(m.group(1)) if m else None
        return out

    # ---------------------------------------------------------------- checks and stress tests
    def verification(self):
        """Does the live model reproduce the numbers the training run saved?"""
        if not self.report:
            return None
        te = self.data.test_mask.numpy()
        c = _confusion(self.y[te], self.probs[te], self.threshold)
        want = self.report["test_confusion_matrix"]
        auc = float(roc_auc_score(self.y[te].astype(int), self.probs[te]))
        return bool(all(c[k.lower()] == want[k] for k in ("TN", "FP", "FN", "TP"))
                    and abs(auc - self.report["test_metrics"]["roc_auc"]) < 1e-6)

    def reality(self):
        if self._reality is not None:
            return self._reality
        d, m = self.data, self.model
        tr, va, te = (d.train_mask.numpy(), d.val_mask.numpy(), d.test_mask.numpy())
        y = self.y
        out = {}
        with torch.no_grad():  # same model with the graph summaries replaced by zeros
            emb = m.encode_nodes(d.x, d.edge_index)
            se, de = torch.zeros_like(emb[d.tx_src]), torch.zeros_like(emb[d.tx_dst])
            z = torch.sigmoid(m.classifier(torch.cat([se, de, se * de, m.edge_encoder(d.tx_edge_attr)], -1)).squeeze(-1)).numpy()
        zs = _confusion(y[te], z[te], self.threshold)
        out["zeroed"] = dict(auc=float(roc_auc_score(y[te].astype(int), z[te])), f1=zs["f1"], rec=zs["recall"], prec=zs["precision"])
        X = d.tx_edge_attr.numpy()
        names = self.meta["feature_names"]

        def baseline(model, cols):
            sc = StandardScaler().fit(X[tr][:, cols])
            model.fit(sc.transform(X[tr][:, cols]), y[tr])
            pv = model.predict_proba(sc.transform(X[va][:, cols]))[:, 1]
            pt = model.predict_proba(sc.transform(X[te][:, cols]))[:, 1]
            th = max(np.arange(.05, .95, .01), key=lambda t: _confusion(y[va], pv, t)["f1"])  # validation only
            s = _confusion(y[te], pt, th)
            return dict(auc=float(roc_auc_score(y[te].astype(int), pt)), f1=s["f1"], rec=s["recall"], prec=s["precision"])

        allc = list(range(len(names)))
        out["logreg"] = baseline(LogisticRegression(max_iter=2000), allc)
        out["histgb"] = baseline(HistGradientBoostingClassifier(random_state=config.RANDOM_SEED), allc)
        noid = [i for i, c in enumerate(names) if c not in ("Sender_Account", "Receiver_Account")]
        out["histgbNoId"] = baseline(HistGradientBoostingClassifier(random_state=config.RANDOM_SEED), noid)
        self._reality = out
        return out

    def facts(self):
        G, meta, m = self.G, self.meta, self.model
        types = {k: sum(1 for _, a in G.nodes(data=True) if a.get("type") == k) for k in PREFIX}
        names = meta["feature_names"]
        return dict(
            payments=int(len(self.ids)), fraud=int(self.y.sum()), nodes=G.number_of_nodes(), nodeTypes=types,
            links=G.number_of_edges(),
            split=dict(train=meta["train_count"], val=meta["val_count"], test=meta["test_count"]),
            params=int(sum(p.numel() for p in m.parameters() if p.requires_grad)),
            featureCount=len(names), idFeatures=[c for c in names if c in ("Sender_Account", "Receiver_Account")],
            arch=dict(nodeIn=m.conv1.in_channels, hidden=m.conv1.out_channels, edgeIn=m.edge_encoder[0].in_features,
                      edgeMid=m.edge_encoder[0].out_features, edgeOut=m.edge_encoder[4].out_features,
                      fuse=m.classifier[0].in_features, c1=m.classifier[0].out_features, c2=m.classifier[3].out_features))

    # ---------------------------------------------------------------- payloads
    def _row(self, i):
        tid = self.ids[i]
        u, v, d = self.tx[tid]
        ft = d.get("fraud_type") if self.y[i] == 1 else None
        g = lambda dct: _node_id(dct[tid]) if tid in dct else None
        return [int(tid), _node_id(u), _node_id(v), g(self.dev), g(self.mer), g(self.upi), round(float(d["amount"]), 2),
                d["channel"], str(d["timestamp"])[:13].replace(" ", "T"), ft, round(float(self.probs[i]), 9), int(self.split[i])]

    def payload(self):
        with self.lock:
            if self._payload is None:
                hist = self.history or {}
                self._payload = dict(
                    cols=["id", "s", "r", "dev", "m", "u", "amt", "ch", "ts", "ft", "p", "sp"],
                    rows=[self._row(i) for i in range(len(self.ids))],
                    history={k: [round(x, 4) for x in v] for k, v in hist.items()},
                    report=self.report, threshold=self.threshold, bestEpoch=int(self.ck.get("epoch", 0)),
                    reality=self.reality(), facts=self.facts(), claims=self.claims,
                    source=dict(generated=datetime.date.today().isoformat(), verified=self.verification(),
                                files=["graph.gpickle", "Cleaned_Database.csv", "outputs/models/best_graphsage.pt",
                                       "outputs/logs/evaluation_report.json", "outputs/logs/training_history.json", "final_report.md"]))
            return self._payload

    # ---------------------------------------------------------------- single payments and metrics
    def payment(self, tid: int):
        i = self.index.get(int(tid))
        if i is None:
            return None
        u, v, d = self.tx[int(tid)]
        p = float(self.probs[i])
        feats = self.data.tx_edge_attr[i].numpy()
        order = np.argsort(-np.abs(feats))[:8]
        label = int(self.y[i])
        split = ["train", "validation", "test"][int(self.split[i])]
        return dict(id=int(tid), sender=u, receiver=v, amount=float(d["amount"]), channel=d["channel"], timestamp=str(d["timestamp"]),
                    device=self.dev.get(int(tid)), merchant=self.mer.get(int(tid)), upi=self.upi.get(int(tid)),
                    fraud=bool(label), fraudType=d.get("fraud_type") if label else None, score=p, threshold=self.threshold,
                    flagged=bool(p >= self.threshold), split=split,
                    topFeatures=[dict(name=self.meta["feature_names"][j], value=float(feats[j])) for j in order])

    def score_new(self, sender: str, receiver: str, features: dict):
        """Score a payment that is not in the data: two existing nodes plus feature values (cleaned, scaled units)."""
        ni = self.ds.node_to_idx
        for n in (sender, receiver):
            if n not in ni:
                raise KeyError(f"Unknown node '{n}'. Use names like account_12.")
        names = self.meta["feature_names"]
        attr = self.col_means.clone()
        for k, val in features.items():
            if k not in names:
                raise KeyError(f"Unknown feature '{k}'.")
            attr[names.index(k)] = float(val)
        with torch.no_grad():
            p = self.model.predict_proba(self.data.x, self.data.edge_index, torch.tensor([ni[sender]]), torch.tensor([ni[receiver]]), attr[None, :])
        p = float(p[0])
        return dict(score=p, threshold=self.threshold, flagged=bool(p >= self.threshold), unspecifiedFeaturesUseDatasetMean=True)

    def metrics_at(self, th: float):
        te = self.data.test_mask.numpy()
        c = _confusion(self.y[te], self.probs[te], th)
        c.update(threshold=th, n=int(te.sum()), roc_auc=float(roc_auc_score(self.y[te].astype(int), self.probs[te])),
                 pr_auc=float(average_precision_score(self.y[te].astype(int), self.probs[te])))
        return c

    def health(self):
        return dict(ok=self.loaded, epoch=int(self.ck.get("epoch", 0)) if self.loaded else None, payments=len(self.ids) if self.loaded else 0,
                    verified=self.verification() if self.loaded else None)
