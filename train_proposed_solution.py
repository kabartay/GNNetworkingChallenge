"""
   Copyright 2025 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import random
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any

# Run eagerly-> Turn true for debugging only
RUN_EAGERLY = False
tf.config.run_functions_eagerly(RUN_EAGERLY)


def _reset_seeds(seed: int = 42) -> None:
    """Reset rng seeds, and also indicate tf if to run eagerly or not

    Parameters
    ----------
    seed : int, optional
        Seed for rngs, by default 42
    """
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def get_default_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Returns the default callbacks for the training of the models
    (EarlyStopping and ReduceLROnPlateau callbacks)
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=0.0001,
            start_from_epoch=4,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            verbose=1,
            mode="min",
            min_delta=0.001,
        ),
    ]


def denorm_MAPE(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean Absolute Percentage Error (MAPE) metric, but denormalized. Specifically,
    it expects both y_true and y_pred to be in log scale, and it returns the MAPE in the
    original scale.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values in log scale.
    y_pred : tf.Tensor
        Predicted values in log scale.

    Returns
    -------
    tf.Tensor
        MAPE in the original scale.
    """
    denorm_y_true = tf.math.exp(y_true)
    denorm_y_pred = tf.math.exp(y_pred)
    return tf.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100


def get_default_hyperparams() -> Dict[str, Any]:
    """Returns the default hyperparameters for the training of the models. That is
    - Adam optimizer with lr=0.00025
    - MeanSquaredError loss (target variables are already transformed with the log,
    so the actual loss is log-MSE).
    - Added denormalized MAPE (Mean Absolute Percentage Error, before the log
    transformation)
    - EarlyStopping and ReduceLROnPlateau callbacks
    - 150 epochs
    """
    return {
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.00025),
        "loss": tf.keras.losses.MeanSquaredError(),
        "metrics": [denorm_MAPE],
        "additional_callbacks": get_default_callbacks(),
        "epochs": 150,
    }


def get_min_max_dict(
    ds: tf.data.Dataset, params: List[str], include_y: Optional[str] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Get the min and the max-min for different parameters of a dataset. Later used by
    the models for the min-max normalization.

    Parameters
    ----------
    ds : tf.data.Dataset
        Training dataset where to base the min-max normalization from.

    params : List[str]
        List of strings indicating the parameters to extract the features from.

    include_y : Optional[str], optional
        Indicates if to also extract the features of the output variable.
        Inputs indicate the string key used on the return dict. If None, it is not
        included.
        By default None.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary containing the values needed for the min-max normalization.
        The first value is the min value of the parameter, and the second is
        1 / (max - min).
    """

    # Use first sample to get the shape of the tensors
    iter_ds = iter(ds)
    sample, label = next(iter_ds)
    params_lists = {param: sample[param].numpy() for param in params}
    if include_y:
        params_lists[include_y] = label.numpy()

    # Include the rest of the samples
    for sample, label in iter_ds:
        for param in params:
            params_lists[param] = np.concatenate(
                (params_lists[param], sample[param].numpy()), axis=0
            )
        if include_y:
            params_lists[include_y] = np.concatenate(
                (params_lists[include_y], label.numpy()), axis=0
            )

    scores = dict()
    for param, param_list in params_lists.items():
        min_val = np.min(param_list, axis=0)
        min_max_val = np.max(param_list, axis=0) - min_val
        if min_max_val.size == 1 and min_max_val == 0:
            scores[param] = [min_val, 0]
            print(f"Min-max normalization Warning: {param} has a max-min of 0.")
        elif min_max_val.size > 1 and np.any(min_max_val == 0):
            min_max_val[min_max_val != 0] = 1 / min_max_val[min_max_val != 0]
            scores[param] = [min_val, min_max_val]
            print(
                f"Min-max normalization Warning: Several values of {param} has a",
                "max-min of 0.",
            )
        else:
            scores[param] = [min_val, 1 / min_max_val]

    return scores


def log_transform(x: dict, y: tf.Tensor) -> Tuple[dict, tf.Tensor]:
    """Apply log transformation to output variable.

    Parameters
    ----------
    x: dict
        Predictor variables
    y: tf.Tensor
        Output variable

    Returns
    -------
    dict
        Predictor variables
    tf.Tensor
        Transformed output variable
    """
    return x, tf.math.log(y)


def train_and_evaluate(
    cbr_mb_ds_path: str,
    mb_ds_path: str,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    metrics: List[tf.keras.metrics.Metric],
    additional_callbacks: List[tf.keras.callbacks.Callback],
    epochs: int = 100,
    ckpt_path: Optional[str] = None,
    tensorboard_path: Optional[str] = None,
    restore_ckpt: bool = False,
    final_eval: bool = True,
) -> Tuple[tf.keras.Model, Union[float, np.ndarray, None]]:
    """
    Train the given model with the given dataset, using the provided parameters
    Besides for defining the hyperparameters, refer to get_default_hyperparams()

    Parameters
    ----------
    cbr_mb_ds_path : str
        Path to the cbr+mb dataset. Datasets are expected to be in tf.data.Dataset
        format, and to be compressed with GZIP.

    model : tf.keras.Model
        Instance of the model to train. Besides being a tf.keras.Model, it should have
        the same constructor and the name parameter as the models in the models module.

    optimizer : tf.keras.Optimizer
        Optimizer used by the training process

    loss : tf.keras.losses.Loss
        Loss function to be used by the process

    metrics : List[tf.keras.metrics.Metric]
        List of additional metrics to consider during training

    additional_callbacks : List[tf.keras.callbacks.Callback], optional
        List containing tensorflow callback functions to be added to the training
        process. A callback to generate tensorboard and checkpoint files at each epoch
        is already added.

    epochs : int, optional
        Number of epochs of in the training process, by default 100

    ckpt_path : Optional[str], optional
        Path where to store the training checkpints, by default
        "{repository root}/ckpt/{model name}"

    tensorboard_path : Optional[str], optional
        Path where to store tensorboard logs, by default
        "{repository root}/tensorboard/{model name}"

    restore_ckpt : bool, optional
        If True, before training the model, it is checked if there is a checkpoint file
        in the ckpt_path. If so, the model loads the latest checkpoint and continues
        training from there. By default False.

    final_eval : bool, optional
        If True, the model is evaluated on the validation dataset one last time after
        training, by default True

    Returns
    -------
    Tuple[tf.keras.Model, Union[float, np.ndarray, None]]
        Instance of the trained model, and the result of its evaluation
    """

    # Reset tf state
    _reset_seeds()
    # Check epoch number is valid
    assert epochs > 0, "Epochs must be greater than 0"
    # Load ds
    ds_train = (
        tf.data.Dataset.load(f"{cbr_mb_ds_path}/training", compression="GZIP")
        .concatenate(tf.data.Dataset.load(f"{mb_ds_path}/training", compression="GZIP"))
        .map(log_transform)
        .shuffle(10000, seed=42, reshuffle_each_iteration=True)
    )
    ds_val = (
        tf.data.Dataset.load(f"{cbr_mb_ds_path}/validation", compression="GZIP")
        .concatenate(
            tf.data.Dataset.load(f"{mb_ds_path}/validation", compression="GZIP")
        )
        .map(log_transform)
    )
    # Checkpoint path
    if ckpt_path is None:
        ckpt_path = f"ckpt/{model.name}"
    # Tensorboard path
    if tensorboard_path is None:
        tensorboard_path = f"tensorboard/{model.name}"

    # Apply min-max normalization
    model.set_min_max_scores(get_min_max_dict(ds_train, model.min_max_scores_fields))

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        run_eagerly=RUN_EAGERLY,
    )
    # Load checkpoint
    if restore_ckpt:
        ckpt = tf.train.latest_checkpoint(ckpt_path)
        if ckpt is not None:
            print("Restoring from checkpoint")
            model.load_weights(ckpt)
        else:
            print(
                f"WARNING: No checkpoint was found at '{ckpt_path}', training from",
                "scratch instead...",
            )
    else:
        print("restore_ckpt = False, training from scratch")

    # Create callbacks
    cpkt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_path, "{epoch:02d}-{val_loss:.4f}"),
        verbose=1,
        mode="min",
        save_best_only=False,
        save_weights_only=True,
        save_freq="epoch",
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_path, histogram_freq=1
    )

    # Train model
    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[cpkt_callback, tensorboard_callback] + additional_callbacks,
        use_multiprocessing=True,
    )

    if final_eval:
        return model, model.evaluate(ds_val)
    else:
        return model, None


if __name__ == "__main__":
    from models import GNN_proposed

    # Prepare variables
    cbr_mb_ds_path = "data/data_cbr_mb_solution"
    mb_ds_path = "data/data_mb_solution"
    _reset_seeds()
    trained_model, evaluation = train_and_evaluate(
        cbr_mb_ds_path, mb_ds_path, GNN_proposed(log=True), **get_default_hyperparams()
    )
    print("Final evaluation:", evaluation)
