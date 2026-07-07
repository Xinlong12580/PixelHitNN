import Trainer

layers = ["L1U", "L1F","L2","L3M","L3P","L4M","L4P"]
axes = ["x","y"]
for layer in layers:
    for axis in axes:
        trainer = Trainer.Trainer(layer=layer, axis=axis)
        #trainer.debug_input()
        #continue
        trainer.prepare_train_test_input()
        trainer.train()
        trainer.test()
        trainer.visualize()
        trainer.plot_otherMethods()
