#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle
import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    obj = obj["model"]

    new_model = {}
    for k, v in obj.items():
        if not k.startswith("module.encoder."):
            continue
        old_k = k
        k = k.replace("module.encoder.", "")
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        new_model[k] = v.numpy()

    res = {
        "model": new_model,
        "__author__": "PixPro",
        "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pickle.dump(res, f)
