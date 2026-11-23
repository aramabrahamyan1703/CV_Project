"""Tests for Set 8: model passports, preprocessing, CNN training and saving.

Passport and image tests need no PyTorch. Model tests need PyTorch and
torchvision (``pip install ultralytics`` brings both) and run on the CPU with
small synthetic images and a randomly initialized InceptionV3, so they never
download anything; the one test of the ImageNet weights runs only if they
are already in torch's cache.
"""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.cnn_data import (
    CIFAR10_CLASSES,
    INCEPTION_V3,
    PASSPORT_FILE,
    SMALL_CNN,
    Passport,
    Preprocessing,
    load_passport,
    save_passport,
    to_dataset_image,
)

HAVE_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision"))
NEEDS_TORCH = unittest.skipUnless(HAVE_TORCH, "needs torch and torchvision (pip install ultralytics)")

if HAVE_TORCH:
    import torch

    from forensics_app.tools import cnn


def _passport(architecture: str = SMALL_CNN) -> Passport:
    pre = Preprocessing(32, 299, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return Passport(
        architecture=architecture,
        classes=CIFAR10_CLASSES,
        preprocessing=pre,
        feature_dim=128,
        dataset={"name": "CIFAR-10", "train_samples": 45000, "val_samples": 5000, "test_samples": 10000},
        history={"loss": [2.0, 1.5], "accuracy": [0.3, 0.5], "val_loss": [1.8, 1.4], "val_accuracy": [0.35, 0.52]},
        test={"loss": 1.4, "accuracy": 0.51},
    )


def _colour_set(count: int, seed: int = 0):
    """Two classes anyone can tell apart: reddish (0) and bluish (1) 32×32 images."""
    rng = np.random.default_rng(seed)
    labels = np.arange(count) % 2
    images = rng.integers(0, 80, size=(count, 32, 32, 3), dtype=np.uint8)
    images[labels == 0, ..., 0] += 150
    images[labels == 1, ..., 2] += 150
    return cnn.ImageSet(images, labels.astype(np.int64))


class PassportTests(unittest.TestCase):
    def test_round_trip_keeps_every_field(self) -> None:
        passport = _passport()
        with TemporaryDirectory() as tmp:
            path = save_passport(passport, Path(tmp) / "model")
            loaded = load_passport(path.parent)
            self.assertEqual(loaded, passport)
            self.assertEqual(load_passport(path.with_name("model.pt")), passport)  # from the weights file
            self.assertEqual(json.loads(path.read_text())["classes"][0], "airplane")

    def test_missing_or_foreign_passports_are_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with self.assertRaisesRegex(ValueError, "No passport.json"):
                load_passport(folder)
            (folder / PASSPORT_FILE).write_text(json.dumps({"classes": ["a"]}))
            with self.assertRaisesRegex(ValueError, "not a model passport"):
                load_passport(folder)
            (folder / PASSPORT_FILE).write_text("{ not json")
            with self.assertRaisesRegex(ValueError, "not a readable passport"):
                load_passport(folder)

    def test_description_names_every_step(self) -> None:
        text = Preprocessing(32, 299, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).describe()
        self.assertIn("32×32", text)
        self.assertIn("bilinear upsampling to 299×299", text)
        self.assertIn("(x − 0.5) / 0.5", text)
        self.assertNotIn("upsampling", Preprocessing(32, 32, (0, 0, 0), (1, 1, 1)).describe())

    def test_any_image_becomes_a_32_pixel_rgb_array(self) -> None:
        gray = to_dataset_image(Image.new("L", (200, 120), 90), 32)
        self.assertEqual((gray.shape, gray.dtype), ((32, 32, 3), np.uint8))
        self.assertTrue((gray == 90).all())
        clear = to_dataset_image(Image.new("RGBA", (64, 64), (255, 0, 0, 0)), 32)
        self.assertTrue((clear == 255).all())  # transparent -> white, not black


@NEEDS_TORCH
class PreprocessingTests(unittest.TestCase):
    def test_small_cnn_input_is_zero_to_one_at_32(self) -> None:
        images = torch.tensor([[[[0, 128, 255]]]], dtype=torch.uint8).expand(1, 32, 32, 3)
        x = cnn.to_input(images, cnn.preprocessing_for(SMALL_CNN))
        self.assertEqual(tuple(x.shape), (1, 3, 32, 32))
        self.assertEqual(x.dtype, torch.float32)
        self.assertAlmostEqual(float(x[0, 0].max()), 0.0)
        self.assertAlmostEqual(float(x[0, 2].min()), 1.0)

    def test_inception_input_is_minus_one_to_one_at_299(self) -> None:
        images = torch.tensor([[[[0, 128, 255]]]], dtype=torch.uint8).expand(2, 32, 32, 3)
        x = cnn.to_input(images, cnn.preprocessing_for(INCEPTION_V3))
        self.assertEqual(tuple(x.shape), (2, 3, 299, 299))
        self.assertAlmostEqual(float(x[:, 0].min()), -1.0, places=6)
        self.assertAlmostEqual(float(x[:, 2].max()), 1.0, places=6)

    def test_unknown_architecture_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            cnn.preprocessing_for("resnet")
        with self.assertRaises(ValueError):
            cnn.build_model("resnet", 10)

    def test_validation_is_the_last_tenth_and_subsets_are_reproducible(self) -> None:
        data = cnn.ImageSet(np.zeros((100, 2, 2, 3), np.uint8), np.arange(100))
        train, val = cnn.split_train_val(data)
        self.assertEqual(list(val.labels), list(range(90, 100)))
        self.assertEqual(len(train), 90)
        small, _ = cnn.split_train_val(data, train_samples=20, seed=3)
        again, _ = cnn.split_train_val(data, train_samples=20, seed=3)
        self.assertEqual(list(small.labels), list(again.labels))
        self.assertTrue(set(small.labels) <= set(range(90)))  # never from validation


@NEEDS_TORCH
class SmallCNNTests(unittest.TestCase):
    def test_features_and_logits_have_the_right_sizes(self) -> None:
        model = cnn.build_model(SMALL_CNN, 10).eval()
        x = torch.zeros(4, 3, 32, 32)
        self.assertEqual(tuple(model.features(x).shape), (4, 128))
        self.assertEqual(tuple(model(x).shape), (4, 10))

    def test_training_learns_two_colours_and_records_history(self) -> None:
        data = _colour_set(96)
        train, val = cnn.split_train_val(data, val_fraction=0.25)
        model = cnn.build_model(SMALL_CNN, 2, seed=0)
        stages = []
        config = cnn.TrainConfig(epochs=3, batch_size=16, learning_rate=0.01)
        result = cnn.train(model, cnn.preprocessing_for(SMALL_CNN), train, val, config, lambda s, d, t: stages.append(s))
        self.assertFalse(result.cancelled)
        self.assertEqual(result.epochs_done, 3)
        self.assertEqual({len(v) for v in result.history.values()}, {3})
        self.assertGreaterEqual(result.history["val_accuracy"][-1], 0.9)
        self.assertIn("Epoch 3/3", stages)

    def test_the_seed_fixes_the_initial_weights(self) -> None:
        first, second = (cnn.build_model(SMALL_CNN, 10, seed=7).state_dict() for _ in range(2))
        self.assertTrue(all(torch.equal(first[k], second[k]) for k in first))
        other = cnn.build_model(SMALL_CNN, 10, seed=8).state_dict()
        self.assertFalse(all(torch.equal(first[k], other[k]) for k in first))

    def test_a_training_run_fills_every_passport_field_of_the_slide(self) -> None:
        data = _colour_set(48)
        train, val = cnn.split_train_val(data, val_fraction=0.25)
        test = _colour_set(10, seed=1)
        pre = cnn.preprocessing_for(SMALL_CNN)
        model = cnn.build_model(SMALL_CNN, 2, seed=0)
        config = cnn.TrainConfig(epochs=2, batch_size=12)
        result = cnn.train(model, pre, train, val, config)
        score = cnn.evaluate(model, pre, test)
        passport = cnn.passport_for(SMALL_CNN, ("red", "blue"), train, val, test, [(config, result)], score)
        with TemporaryDirectory() as tmp:
            cnn.save_model(model, passport, Path(tmp))
            loaded = load_passport(Path(tmp))
        self.assertEqual(loaded, passport)
        # Slide 8.1: dataset, sample size, class names, preprocessing, training history
        self.assertEqual(loaded.dataset["name"], "CIFAR-10")
        self.assertEqual(
            (loaded.dataset["train_samples"], loaded.dataset["val_samples"], loaded.dataset["test_samples"]), (36, 12, 10)
        )
        self.assertEqual(loaded.classes, ("red", "blue"))
        self.assertEqual(loaded.preprocessing, pre)
        self.assertEqual(set(loaded.history), {"loss", "accuracy", "val_loss", "val_accuracy"})
        self.assertTrue(all(len(values) == 2 for values in loaded.history.values()))
        self.assertEqual(loaded.test["accuracy"], score[1])
        self.assertEqual(loaded.training["epochs_done"], 2)
        self.assertIn("PyTorch", loaded.training["framework"])

    def test_cancel_keeps_finished_epochs_or_raises_before_the_first(self) -> None:
        train, val = cnn.split_train_val(_colour_set(40), val_fraction=0.25)
        pre = cnn.preprocessing_for(SMALL_CNN)
        config = cnn.TrainConfig(epochs=5, batch_size=10)
        with self.assertRaises(cnn.TrainingCancelled):
            cnn.train(cnn.build_model(SMALL_CNN, 2), pre, train, val, config, cancelled=lambda: True)
        calls = iter(range(10_000))
        # 3 training batches + 1 validation batch per epoch: stop during epoch 2
        result = cnn.train(cnn.build_model(SMALL_CNN, 2), pre, train, val, config, cancelled=lambda: next(calls) >= 5)
        self.assertTrue(result.cancelled)
        self.assertEqual(result.epochs_done, 1)
        self.assertEqual(len(result.history["loss"]), 1)

    def test_cancelled_weights_are_those_of_the_last_finished_epoch(self) -> None:
        train, val = cnn.split_train_val(_colour_set(40), val_fraction=0.25)
        pre = cnn.preprocessing_for(SMALL_CNN)
        untouched = cnn.build_model(SMALL_CNN, 2, seed=1)
        start = {k: v.clone() for k, v in untouched.state_dict().items()}
        with self.assertRaises(cnn.TrainingCancelled):  # stopped in epoch 1: back to the start
            calls = iter(range(10_000))
            cnn.train(untouched, pre, train, val, cnn.TrainConfig(epochs=5, batch_size=10),
                      cancelled=lambda: next(calls) >= 2)
        self.assertTrue(all(torch.equal(v, start[k]) for k, v in untouched.state_dict().items()))

        one_epoch = cnn.build_model(SMALL_CNN, 2, seed=1)
        cnn.train(one_epoch, pre, train, val, cnn.TrainConfig(epochs=1, batch_size=10))
        stopped = cnn.build_model(SMALL_CNN, 2, seed=1)
        calls = iter(range(10_000))  # 4 checks per epoch: stop 2 batches into epoch 2
        cnn.train(stopped, pre, train, val, cnn.TrainConfig(epochs=5, batch_size=10), cancelled=lambda: next(calls) >= 6)
        expected = one_epoch.state_dict()
        self.assertTrue(all(torch.equal(v, expected[k]) for k, v in stopped.state_dict().items()))

    def test_saved_model_loads_with_identical_predictions(self) -> None:
        data = _colour_set(8)
        pre = cnn.preprocessing_for(SMALL_CNN)
        model = cnn.build_model(SMALL_CNN, 2)
        before = cnn.predict_proba(model, pre, data.images)
        np.testing.assert_allclose(before.sum(axis=1), 1.0, rtol=1e-5)
        passport = Passport(SMALL_CNN, ("red", "blue"), pre, model.feature_dim)
        with TemporaryDirectory() as tmp:
            cnn.save_model(model, passport, Path(tmp))
            loaded, loaded_passport = cnn.load_model(Path(tmp) / "model.pt")
        self.assertEqual(loaded_passport.classes, ("red", "blue"))
        np.testing.assert_allclose(cnn.predict_proba(loaded, pre, data.images), before, rtol=1e-5, atol=1e-6)
        features = cnn.extract_features(loaded, pre, data.images, batch_size=3)
        self.assertEqual(features.shape, (8, 128))
        self.assertEqual(features.dtype, np.float32)

    def test_weights_of_another_architecture_or_no_weights_are_refused(self) -> None:
        pre = cnn.preprocessing_for(SMALL_CNN)
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            save_passport(Passport(SMALL_CNN, CIFAR10_CLASSES, pre, 128), folder)
            with self.assertRaisesRegex(ValueError, "No model.pt"):
                cnn.load_model(folder)
            cnn.save_model(cnn.build_model(SMALL_CNN, 3), Passport(SMALL_CNN, CIFAR10_CLASSES, pre, 128), folder)
            with self.assertRaisesRegex(ValueError, "does not hold weights"):
                cnn.load_model(folder)  # 3 outputs saved, passport lists 10 classes


@NEEDS_TORCH
class InceptionTests(unittest.TestCase):
    def test_built_and_loaded_models_agree_on_input_handling(self) -> None:
        model = cnn.build_model(INCEPTION_V3, 10, pretrained=False)
        net = model.features
        self.assertFalse(net.transform_input)  # input scaling lives in the passport alone
        self.assertIsNone(net.AuxLogits)
        self.assertEqual(cnn.preprocessing_for(INCEPTION_V3).mean, (0.5, 0.5, 0.5))
        images = _colour_set(2).images
        pre = cnn.preprocessing_for(INCEPTION_V3)
        before = cnn.predict_proba(model, pre, images)
        with TemporaryDirectory() as tmp:
            cnn.save_model(model, Passport(INCEPTION_V3, CIFAR10_CLASSES, pre, 2048), Path(tmp))
            loaded, _ = cnn.load_model(Path(tmp))
        self.assertFalse(loaded.features.transform_input)
        np.testing.assert_allclose(cnn.predict_proba(loaded, pre, images), before, rtol=1e-4, atol=1e-6)
        self.assertEqual(cnn.extract_features(loaded, pre, images).shape, (2, 2048))

    def test_frozen_backbone_keeps_its_weights_and_batchnorm_statistics(self) -> None:
        torch.manual_seed(0)
        model = cnn.build_model(INCEPTION_V3, 2, pretrained=False)
        backbone = {k: v.clone() for k, v in model.features.state_dict().items()}
        head = {k: v.clone() for k, v in model.classifier.state_dict().items()}
        train, val = cnn.split_train_val(_colour_set(8), val_fraction=0.25)
        config = cnn.TrainConfig(epochs=1, batch_size=4, optimizer="adam", learning_rate=1e-3, freeze_backbone=True)
        cnn.train(model, cnn.preprocessing_for(INCEPTION_V3), train, val, config)
        for key, value in model.features.state_dict().items():  # includes running_mean / running_var
            self.assertTrue(torch.equal(value, backbone[key]), key)
        changed = [not torch.equal(v, head[k]) for k, v in model.classifier.state_dict().items()]
        self.assertTrue(any(changed))


class TrainingToolTests(unittest.TestCase):
    """The CNN training tool's job, figure and estimate (Set 8.1)."""

    def _request(self, folder: Path, epochs: int = 2, architecture: str = SMALL_CNN, fine_tune: int = 0):
        from forensics_app.tools.cnn_training import CnnTrainRequest

        return CnnTrainRequest(architecture, epochs, fine_tune, 1_000, "cpu", folder)

    def test_estimate_grows_with_epochs_and_is_only_given_for_mps(self) -> None:
        from forensics_app.tools.cnn_training import estimate_seconds

        one = replace(self._request(Path("m"), 1), device="mps")
        self.assertAlmostEqual(estimate_seconds(replace(one, epochs=4)), 4 * estimate_seconds(one))
        head = replace(one, architecture=INCEPTION_V3)
        self.assertGreater(estimate_seconds(replace(head, fine_tune_epochs=1)), estimate_seconds(head))
        self.assertIsNone(estimate_seconds(self._request(Path("m"))))  # cpu: not measured

    def test_the_dialog_defaults_are_valid(self) -> None:
        from forensics_app.tools.cnn_training import TRAIN_FIELDS
        from forensics_app.tools.param_dialog import parse_int_fields

        self.assertEqual(parse_int_fields(TRAIN_FIELDS, tuple(str(f.default) for f in TRAIN_FIELDS)), (10, 0, 45_000))

    def test_history_figure_from_a_passport(self) -> None:
        from forensics_app.tools.cnn_training import render_history

        passport = _passport()
        passport.training = {"stages": [{"epochs_done": 1}, {"epochs_done": 1}]}  # with a fine-tuning line
        image = render_history(passport, "run")
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(render_history(Passport(SMALL_CNN, ("a",), passport.preprocessing, 1)).mode, "RGB")

    @NEEDS_TORCH
    def test_job_trains_tests_and_saves_a_loadable_model(self) -> None:
        from forensics_app.tools.cnn_training import CnnTrainingJob

        data = lambda: (_colour_set(60), _colour_set(20, seed=1), ("red", "blue"))  # noqa: E731
        with TemporaryDirectory() as tmp:
            job = CnnTrainingJob(self._request(Path(tmp) / "small"), load_data=data)
            job.start()
            job.join(120)
            self.assertIsNone(job.error, repr(job.error))
            outcome = job.outcome
            self.assertEqual(outcome.passport.training["epochs_done"], 2)
            self.assertEqual(outcome.passport.dataset["test_samples"], 20)
            self.assertIn("accuracy", outcome.passport.test)
            model, passport = cnn.load_model(outcome.folder)
            self.assertEqual(passport, outcome.passport)
            self.assertEqual(passport.classes, ("red", "blue"))
            done, total, text = job.status()
            self.assertEqual(done, total)
            self.assertIn("Testing", text)

    @NEEDS_TORCH
    def test_stop_before_the_first_epoch_saves_nothing(self) -> None:
        from forensics_app.tools.cnn_training import CnnTrainingJob

        data = lambda: (_colour_set(60), _colour_set(20, seed=1), ("red", "blue"))  # noqa: E731
        with TemporaryDirectory() as tmp:
            job = CnnTrainingJob(self._request(Path(tmp) / "small", epochs=3), load_data=data)
            job.request_stop()
            job.start()
            job.join(120)
            self.assertIsNone(job.error, repr(job.error))
            self.assertIsNone(job.outcome)
            self.assertFalse((Path(tmp) / "small").exists())


class ClassifyToolTests(unittest.TestCase):
    """The CNN classification tool: passport display and classifying an image (Set 8.2)."""

    def test_passport_rows_cover_the_slide(self) -> None:
        from forensics_app.tools.cnn_classify import passport_details

        passport = _passport(INCEPTION_V3)
        passport.training = {
            "stages": [{"epochs": 2, "epochs_done": 2, "optimizer": "adam", "learning_rate": 0.001,
                        "frozen_backbone": True, "batch_size": 64, "device": "mps", "seconds": 90}],
            "framework": "PyTorch 2",
        }
        rows = passport_details(passport)
        self.assertIn("InceptionV3", rows["Model"])
        self.assertEqual(rows["Samples"], "45,000 train · 5,000 validation · 10,000 test")
        self.assertTrue(rows["Classes"].startswith("airplane, automobile"))
        self.assertIn("299×299", rows["Preprocessing"])
        self.assertIn("head only", rows["Training 1"])
        self.assertIn("2 epochs", rows["History"])
        self.assertEqual(rows["Test accuracy"], "51.0% (loss 1.400)")

    def test_a_model_without_passport_is_described_not_raised(self) -> None:
        from forensics_app.tools.cnn_classify import describe_weights

        with TemporaryDirectory() as tmp:
            self.assertIn("No passport.json", describe_weights(Path(tmp) / "model.pt"))

    @NEEDS_TORCH
    def test_classifying_uses_the_training_preprocessing(self) -> None:
        from forensics_app.tools import cnn_classify as cc

        data = _colour_set(4)
        pre = cnn.preprocessing_for(SMALL_CNN)
        model = cnn.build_model(SMALL_CNN, 2, seed=0)
        passport = Passport(SMALL_CNN, ("red", "blue"), pre, 128, history=_passport().history)
        with TemporaryDirectory() as tmp:
            weights = cnn.save_model(model, passport, Path(tmp))
            loaded, loaded_passport = cc.load_cached(weights, "cpu")
            self.assertIs(cc.load_cached(weights, "cpu")[0], loaded)  # not reloaded
            image = Image.fromarray(data.images[1])
            result = cc.classify(loaded, loaded_passport, image)
            direct = cnn.predict_proba(model, pre, data.images[1:2])[0]
            self.assertAlmostEqual(result.ranking[0][1], float(direct.max()), places=5)
            self.assertEqual(result.ranking[0][0], ("red", "blue")[int(direct.argmax())])
            self.assertEqual([p for _, p in result.ranking], sorted((p for _, p in result.ranking), reverse=True))
            self.assertTrue(np.array_equal(result.seen, data.images[1]))  # 32×32 already: untouched
            big = cc.classify(loaded, loaded_passport, image.resize((300, 200)))
            self.assertEqual(big.seen.shape, (32, 32, 3))
            self.assertEqual(cc.render_classification(image, result, loaded_passport, 2).mode, "RGB")

            tool = cc.CnnClassifyTool()
            document = ImageDocument()
            document.current = image.resize((128, 128))
            tool._ask_params = lambda _parent: None
            self.assertIsNone(tool.run(None, document))
            tool._ask_params = lambda _parent: cc.ClassifyParams(weights, 2)
            outcome = tool.run(None, document)
        self.assertIn("1. ", next(k for k in outcome.details if k.startswith("1.")))
        self.assertEqual(outcome.details["Classes"], "red, blue")
        self.assertEqual(outcome.image.mode, "RGB")


def _clusters(per_class: int = 60, dims: int = 64, seed: int = 0):
    """Three well-separated Gaussian blobs of feature vectors."""
    from forensics_app.tools.cnn_data import FeatureSet

    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 10, size=(3, dims))
    labels = np.repeat(np.arange(3), per_class)
    features = (centres[labels] + rng.normal(0, 1, size=(len(labels), dims))).astype(np.float32)
    return FeatureSet(features, labels.astype(np.int64), ("a", "b", "c"), SMALL_CNN, "model.pt")


class FeatureTests(unittest.TestCase):
    """Saving feature vectors and projecting them with t-SNE and UMAP (Set 8.3)."""

    def test_feature_files_round_trip_and_bad_ones_are_refused(self) -> None:
        from forensics_app.tools.cnn_data import load_features, save_features

        features = _clusters(5)
        with TemporaryDirectory() as tmp:
            path = save_features(features, Path(tmp) / "f" / "small_cnn_test.npz")
            loaded = load_features(path)
            np.testing.assert_array_equal(loaded.features, features.features)
            np.testing.assert_array_equal(loaded.labels, features.labels)
            self.assertEqual(
                (loaded.classes, loaded.architecture, loaded.split, loaded.layer), (("a", "b", "c"), SMALL_CNN, "test", "features")
            )
            np.savez(Path(tmp) / "other.npz", features=np.zeros((3, 2)))
            with self.assertRaisesRegex(ValueError, "missing"):
                load_features(Path(tmp) / "other.npz")
            np.savez(Path(tmp) / "bad.npz", features=np.zeros((3, 2)), labels=np.array([0, 1, 5]),
                     classes=np.array(["a", "b"]), architecture=np.array("x"))
            with self.assertRaisesRegex(ValueError, "outside"):
                load_features(Path(tmp) / "bad.npz")
            (Path(tmp) / "text.npz").write_text("nope")
            with self.assertRaisesRegex(ValueError, "not a readable"):
                load_features(Path(tmp) / "text.npz")

    def test_subsets_are_seeded_and_knn_accuracy_measures_separation(self) -> None:
        from forensics_app.tools import cnn_features as cf

        a, b = cf.choose_points(50, 200), cf.choose_points(50, 200)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(a)), 50)
        np.testing.assert_array_equal(cf.choose_points(500, 200), np.arange(200))
        blobs = _clusters()
        self.assertEqual(cf.knn_accuracy(blobs.features, blobs.labels), 1.0)
        shuffled = np.random.default_rng(1).permutation(blobs.labels)
        self.assertLess(cf.knn_accuracy(blobs.features, shuffled), 0.6)

    def test_pca_keeps_small_vectors_and_shrinks_large_ones(self) -> None:
        from forensics_app.tools.cnn_features import pca_reduce

        blobs = _clusters(dims=64)
        reduced, kept = pca_reduce(blobs.features, 10)
        self.assertEqual(reduced.shape, (180, 10))
        self.assertGreater(kept, 0.5)
        same, all_kept = pca_reduce(blobs.features[:, :8], 10)
        self.assertEqual((same.shape, all_kept), ((180, 8), 1.0))

    @unittest.skipUnless(importlib.util.find_spec("umap") is not None, "needs umap-learn")
    def test_both_projections_keep_well_separated_classes_apart(self) -> None:
        from forensics_app.tools import cnn_features as cf

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.npz"
            params = cf.ProjectParams(path, 150, 30, 15)
            result = cf.project(_clusters(), params)
            self.assertEqual(len(result.labels), 150)
            self.assertEqual(result.reduced_dims, 50)
            for projection in result.projections:
                self.assertEqual(projection.points.shape, (150, 2))
                self.assertGreater(projection.knn_accuracy, 0.95, projection.name)
            again = cf.project(_clusters(), params)  # seeded: the same picture
            np.testing.assert_allclose(again.projections[0].points, result.projections[0].points, rtol=1e-4, atol=1e-4)
            self.assertEqual(cf.render_projections(result, ("a", "b", "c"), "t").mode, "RGB")

            from forensics_app.tools.cnn_data import save_features

            save_features(_clusters(), path)
            tool = cf.FeatureProjectionTool()
            outcome = tool._result(params, cf.load_features(path), result)
        self.assertIn("t-SNE", outcome.message)
        self.assertEqual(outcome.details["Points"], "150 of 180 (seeded random subset)")

    @NEEDS_TORCH
    def test_extraction_gives_one_vector_per_test_image(self) -> None:
        from forensics_app.tools import cnn_features as cf

        pre = cnn.preprocessing_for(SMALL_CNN)
        model = cnn.build_model(SMALL_CNN, 2, seed=0)
        test = _colour_set(30)
        with TemporaryDirectory() as tmp:
            weights = cnn.save_model(model, Passport(SMALL_CNN, ("red", "blue"), pre, 128), Path(tmp))
            features = cf.extract_test_features(weights, 20, "cpu", load_test=lambda: test)
            self.assertEqual(features.features.shape, (20, 128))
            np.testing.assert_array_equal(features.labels, test.labels[:20])
            self.assertEqual(features.classes, ("red", "blue"))
            np.testing.assert_allclose(features.features, cnn.extract_features(model, pre, test.images[:20]), rtol=1e-5)
            stopped = cf.extract_test_features(weights, 20, "cpu", cancelled=lambda: True, load_test=lambda: test)
            self.assertIsNone(stopped)
            with self.assertRaisesRegex(ValueError, "no 'head' layer"):  # the small CNN's head is one layer
                cf.extract_test_features(weights, 5, "cpu", load_test=lambda: test, layer="head")
        self.assertEqual(cf.layer_for(SMALL_CNN, True), "features")
        self.assertEqual(cf.layer_for(INCEPTION_V3, True), "head")

    @NEEDS_TORCH
    def test_inception_head_layer_is_the_trained_1024_unit_layer(self) -> None:
        from forensics_app.tools.cnn_data import features_path

        model = cnn.build_model(INCEPTION_V3, 10, pretrained=False, seed=0)
        pre = cnn.preprocessing_for(INCEPTION_V3)
        images = _colour_set(2).images
        head = cnn.extract_features(model, pre, images, layer="head")
        backbone = cnn.extract_features(model, pre, images)
        self.assertEqual((head.shape, backbone.shape), ((2, 1024), (2, 2048)))
        with torch.no_grad():
            expected = model.classifier[:-1](torch.from_numpy(backbone)).numpy()
        np.testing.assert_allclose(head, expected, rtol=1e-4, atol=1e-5)
        self.assertEqual(features_path(INCEPTION_V3, layer="head").name, "inception_v3_head_test.npz")
        self.assertEqual(features_path(INCEPTION_V3).name, "inception_v3_test.npz")


def _imagenet_weights_cached() -> bool:
    if not HAVE_TORCH:
        return False
    from torchvision.models import Inception_V3_Weights

    name = Path(Inception_V3_Weights.IMAGENET1K_V1.url).name
    return (Path(torch.hub.get_dir()) / "checkpoints" / name).is_file()


@unittest.skipUnless(_imagenet_weights_cached(), "InceptionV3 ImageNet weights not in torch's cache")
class PretrainedInceptionTests(unittest.TestCase):
    def test_pretrained_build_matches_the_load_path(self) -> None:
        pretrained = cnn.build_model(INCEPTION_V3, 10, pretrained=True)
        blank = cnn.build_model(INCEPTION_V3, 10, pretrained=False)
        self.assertEqual(pretrained.features.transform_input, blank.features.transform_input)
        self.assertIsNone(pretrained.features.AuxLogits)
        self.assertEqual(set(pretrained.state_dict()), set(blank.state_dict()))


if __name__ == "__main__":
    unittest.main()
