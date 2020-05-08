#!/usr/bin/env python

import json
from os.path import join, dirname, abspath, isfile
import math
import traceback

import click
import numpy as np

from rastervision2.core import Predictor
from rastervision2.pipeline import rv_config, Verbosity

from integration_tests.chip_classification_tests.experiment \
    import ChipClassificationIntegrationTest
from integration_tests.object_detection_tests.experiment \
    import ObjectDetectionIntegrationTest
from integration_tests.semantic_segmentation_tests.experiment \
    import SemanticSegmentationIntegrationTest

chip_classification = 'chip_classification'
object_detection = 'object_detection'
semantic_segmentation = 'semantic_segmentation'
all_tests = [chip_classification, object_detection, semantic_segmentation]
TEST_ROOT_DIR = dirname(abspath(__file__))

np.random.seed(1234)


def console_info(msg):
    click.echo(click.style(msg, fg='green'))


def console_warning(msg):
    click.echo(click.style(msg, fg='yellow'))


def console_error(msg):
    click.echo(click.style(msg, fg='red', err=True))


class TestError():
    def __init__(self, test, message, details=None):
        self.test = test
        self.message = message
        self.details = details

    def __str__(self):
        return ('Error\n' + '------\n' + 'Test: {}\n'.format(self.test) +
                'Message: {}\n'.format(self.message) + 'Details: {}'.format(
                    str(self.details)) if self.details else '' + '\n')


def get_test_dir(test):
    return join(TEST_ROOT_DIR, test.lower().replace('-', '_'))


def get_expected_eval_path(test):
    return join('{}_tests'.format(get_test_dir(test)),
                        'expected-output/eval.json')


def get_actual_eval_path(test, tmp_dir):
    return join(tmp_dir, test.lower(), 'eval/eval.json')


def open_json(path):
    with open(path, 'r') as file:
        return json.load(file)


def check_eval_item(test, expected_item, actual_item):
    errors = []
    f1_threshold = 0.01
    class_name = expected_item['class_name']

    expected_f1 = expected_item['f1'] or 0.0
    actual_f1 = actual_item['f1'] or 0.0
    if math.fabs(expected_f1 - actual_f1) > f1_threshold:
        errors.append(
            TestError(
                test, 'F1 scores are not close enough',
                'for class_name: {} expected f1: {}, actual f1: {}'.format(
                    class_name, expected_item['f1'], actual_item['f1'])))

    return errors


def check_eval(test, tmp_dir):
    errors = []

    actual_eval_path = get_actual_eval_path(test, tmp_dir)
    expected_eval_path = get_expected_eval_path(test)

    if isfile(actual_eval_path):
        expected_eval = open_json(expected_eval_path)['overall']
        actual_eval = open_json(actual_eval_path)['overall']

        for expected_item in expected_eval:
            class_name = expected_item['class_name']
            actual_item = \
                next(filter(
                    lambda x: x['class_name'] == class_name, actual_eval))
            errors.extend(check_eval_item(test, expected_item, actual_item))
    else:
        errors.append(
            TestError(test, 'actual eval file does not exist',
                      actual_eval_path))

    return errors


def get_pipeline(test, tmp_dir):
    if test == object_detection:
        return ObjectDetectionIntegrationTest().exp_main(
            join(tmp_dir, test.lower()))
    if test == chip_classification:
        return ChipClassificationIntegrationTest().exp_main(
            join(tmp_dir, test.lower()))
    if test == semantic_segmentation:
        return SemanticSegmentationIntegrationTest().exp_main(
            join(tmp_dir, test.lower()))

    raise Exception('Unknown test {}'.format(test))


def test_model_bundle_validation(pipeline, test, tmp_dir, image_uri):
    console_info('Checking predict command validation...')
    errors = []
    pp = pipeline.task.model_bundle_uri
    predict = Predictor(pp, tmp_dir, channel_order=[0, 1, 7]).predict
    try:
        predict(image_uri, 'x.txt')
        e = TestError(test,
                      ('Predictor should have raised exception due to invalid '
                       'channel_order, but did not.'),
                      'in pipeline {}'.format(pipeline.id))
        errors.append(e)
    except ValueError:
        pass

    return errors


def test_model_bundle_results(pipeline, test, tmp_dir, scenes,
                                    scenes_to_uris):
    console_info('Checking predict package produces same results...')
    errors = []
    pp = pipeline.task.model_bundle_uri
    predict = Predictor(pp, tmp_dir).predict

    for scene_config in scenes:
        # Need to write out labels and read them back,
        # otherwise the floating point precision direct box
        # coordinates will not match those from the PREDICT
        # command, which are rounded to pixel coordinates
        # via pyproj logic (in the case of rasterio crs transformer.
        predictor_label_store_uri = join(
            tmp_dir, test.lower(), 'predictor/{}'.format(scene_config.id))
        uri = scenes_to_uris[scene_config.id]

        predict(uri, predictor_label_store_uri)

        scene = scene_config.create_scene(pipeline.task, tmp_dir)

        scene_labels = scene.prediction_label_store.get_labels()

        extent = scene.raster_source.get_extent()
        crs_transformer = scene.raster_source.get_crs_transformer()
        predictor_label_store = scene_config.label_store \
                                       .for_prediction(
                                           predictor_label_store_uri) \
                                       .create_store(
                                           pipeline.task,
                                           extent,
                                           crs_transformer,
                                           tmp_dir)

        from rastervision.data import ActivateMixin
        with ActivateMixin.compose(scene, predictor_label_store):
            if not predictor_label_store.get_labels() == scene_labels:
                e = TestError(
                    test, ('Predictor did not produce the same labels '
                           'as the Predict command'),
                    'for scene {} in pipeline {}'.format(
                        scene_config.id, pipeline.id))
                errors.append(e)

    return errors


def test_model_bundle(pipeline,
                            test,
                            tmp_dir,
                            check_channel_order=False):
    # Check the prediction package
    # This will only work with raster_sources that
    # have a single URI.
    skip = False
    errors = []
    pipeline = pipeline.fully_resolve()

    scenes_to_uris = {}
    scenes = pipeline.dataset.validation_scenes
    for scene in scenes:
        rs = scene.raster_source
        if hasattr(rs, 'uri'):
            scenes_to_uris[scene.id] = rs.uri
        elif hasattr(rs, 'uris'):
            uris = rs.uris
            if len(uris) > 1:
                skip = True
            else:
                scenes_to_uris[scene.id] = uris[0]
        else:
            skip = True

    if skip:
        console_warning('Skipping predict package test for '
                        'test {}, pipeline {}'.format(test, pipeline.id))
    else:
        if check_channel_order:
            errors.extend(
                test_model_bundle_validation(pipeline, test, tmp_dir,
                                                   uris[0]))
        else:
            errors.extend(
                test_model_bundle_results(pipeline, test, tmp_dir,
                                                scenes, scenes_to_uris))

    return errors


def run_test(test, use_tf, tmp_dir):
    errors = []
    pipeline = get_pipeline(test, use_tf, tmp_dir)

    # Check serialization
    mb_uri = join(pipeline.bundle_uri, 'model_bundle.zip')
    pipeline.task.model_bundle_uri = mb_uri
    msg = pipeline.to_proto()
    pipeline = rv.pipelineConfig.from_proto(msg)

    # Check that running doesn't raise any exceptions.
    try:
        IntegrationTestpipelineRunner(join(tmp_dir, test.lower())) \
            .run(pipeline, rerun_commands=True, splits=2,
                 commands_to_run=commands_to_run)

    except Exception:
        errors.append(
            TestError(test, 'raised an exception while running',
                      traceback.format_exc()))
        return errors

    # Check that the eval is similar to expected eval.
    errors.extend(check_eval(test, tmp_dir))

    if not errors:
        errors.extend(test_model_bundle(pipeline, test, tmp_dir))
        errors.extend(
            test_model_bundle(
                pipeline, test, tmp_dir, check_channel_order=True))

    return errors


@click.command()
@click.argument('tests', nargs=-1)
@click.option(
    '--root-uri',
    '-t',
    help=('Sets the rv_root directory used. '
          'If set, test will not clean this directory up.'))
@click.option(
    '--verbose', '-v', is_flag=True, help=('Sets the logging level to DEBUG.'))
def main(tests, root_uri, verbose):
    """Runs RV end-to-end and checks that evaluation metrics are correct."""
    if len(tests) == 0:
        tests = all_tests

    if verbose:
        rv_config.set(verbosity=Verbosity.DEBUG)

    with rv_config.get_tmp_dir() as tmp_dir:
        if root_uri:
            tmp_dir = root_uri

        errors = []
        for test in tests:
            if test not in all_tests:
                print('{} is not a valid test.'.format(test))
                return

            errors.extend(run_test(test, tmp_dir))

            for error in errors:
                print(error)

        for test in tests:
            nb_test_errors = len(
                list(filter(lambda error: error.test == test, errors)))
            if nb_test_errors == 0:
                print('{} test passed!'.format(test))

        if errors:
            exit(1)


if __name__ == '__main__':
    main()
