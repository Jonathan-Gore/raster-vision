import os
from os.path import join

from rastervision2.core.rv_pipeline import *
from rastervision2.core.backend import *
from rastervision2.core.data import *
from rastervision2.core.analyzer import *
from rastervision2.pytorch_backend import *
from rastervision2.pytorch_learner import *
from rastervision2.examples.utils import get_scene_info, save_image_crop


def get_config(runner, root_uri, data_uri=None, full_train=False):
    def get_path(part):
        if full_train:
            return os.path.join(data_uri, part)
        else:
            return os.path.join(os.path.dirname(__file__), part)

    class_config = ClassConfig(
        names=['car', 'building', 'background'],
        colors=['red', 'blue', 'black'])

    if full_train:
        model = ClassificationModelConfig(backbone=Backbone.resnet50)
        solver = SolverConfig(
            lr=1e-4, num_epochs=10, test_num_epochs=3, batch_sz=16, one_cycle=True,
            sync_interval=200)
    else:
        pretrained_uri = (
            'https://github.com/azavea/raster-vision-data/releases/download/'
            'v0.9.0/pytorch_chip_classification_test.pth')
        model = ClassificationModelConfig(
            backbone=Backbone.resnet50, init_weights=pretrained_uri)
        solver = SolverConfig(
            lr=1e-9, num_epochs=1, test_num_epochs=1, batch_sz=2, one_cycle=False,
            sync_interval=200)
    backend = PyTorchChipClassificationConfig(
        model=model,
        solver=solver,
        log_tensorboard=False,
        run_tensorboard=False)

    def make_scene(img_path, label_path):
        id = os.path.basename(img_path))
        label_source = ChipClassificationLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=label_path, ignore_crs_field=True),
            ioa_thresh=0.5,
            use_intersection_over_cell=False,
            pick_min_class_id=True,
            background_class_id=2,
            infer_cells=True)

        raster_source = RasterioSourceConfig(
            channel_order=[0, 1, 2], uris=[img_path],
            transformers=[StatsTransformerConfig()])

        return SceneConfig(
            id=id,
            raster_source=raster_source,
            label_source=label_source)

    img_path = get_path('scene/image.tif')
    label_path = get_path('scene/labels.json')

    img2_path = get_path('scene/image2.tif')
    label2_path = get_path('scene/labels2.json')

    train_scenes = [make_scene(img_path, label_path)]
    val_scenes = [make_scene(img2_path, label2_path)]
    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=val_scenes)

    config = ChipClassificationConfig(
        root_uri=root_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=200,
        debug=True)

    return config
