common = {
    "pitch_x" : 150.0, # microns
    "pitch_y" : 100.0, # microns
    "batch_size" : 4096,
    "epochs" : 200,
    "loss_name": "nll_loss",
    "input_file": "test_clusters.root",
    "debug_predictions": False,
    }

layer_configs = {
    "L1U" : {
        "checkpoint_x" : "checkpoints/L1U_x.weights.h5",
        "checkpoint_y" : "checkpoints/L1U_y.weights.h5",
        "model_dest_x": "checkpoints/L1U_x.keras",
        "model_dest_y": "checkpoints/L1U_y.keras",
    }
}

# Filter rules for each layer/module type (BPIX only for now).
# Each entry is a lambda that accepts arrays (Layer, Ladder, Module)
# and returns a boolean mask selecting the matching clusters.
layer_filter_rules = {
    "L1U": lambda Layer, Ladder, Module: (Layer == 1) & (Ladder % 2 == 1),
    "L1F": lambda Layer, Ladder, Module: (Layer == 1) & (Ladder % 2 == 0),
    "L2":  lambda Layer, Ladder, Module: (Layer == 2),
    "L3M": lambda Layer, Ladder, Module: (Layer == 3) & (Module <= 4),
    "L3P": lambda Layer, Ladder, Module: (Layer == 3) & (Module >= 5),
    "L4M": lambda Layer, Ladder, Module: (Layer == 4) & (Module <= 4),
    "L4P": lambda Layer, Ladder, Module: (Layer == 4) & (Module >= 5),
}