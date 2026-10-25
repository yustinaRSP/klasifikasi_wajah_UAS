"""Pipeline training. Jalankan dari root proyek:  python -m src.training.train"""
from tensorflow import keras

import config
from src.data.loader import compute_class_weights, load_datasets, load_test_dataset
from src.evaluation.evaluate import evaluate_model, plot_history
from src.models.builder import build_model, compile_model, unfreeze_last_block


def _callbacks(initial_best=None):
    return [
        keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_accuracy", save_best_only=True,
            initial_value_threshold=initial_best),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]


def main():
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = load_datasets()
    class_weight = compute_class_weights()
    model = build_model()

    print("\n### Fase 1: latih head (VGG16 dibekukan)")
    compile_model(model, config.LR_HEAD)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS_HEAD,
                   class_weight=class_weight, callbacks=_callbacks())

    print("\n### Fase 2: fine-tune block5")
    unfreeze_last_block(model)
    compile_model(model, config.LR_FINE)      # wajib compile ulang
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS_FINE,
                   class_weight=class_weight,
                   callbacks=_callbacks(max(h1.history["val_accuracy"])))

    history = {k: h1.history[k] + h2.history[k]
               for k in ("accuracy", "val_accuracy", "loss", "val_loss")}
    plot_history(history, split_epoch=len(h1.history["loss"]))

    best = keras.models.load_model(config.MODEL_PATH)
    test_ds, paths = load_test_dataset()
    evaluate_model(best, test_ds, paths)
    print(f"\nModel tersimpan di: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
