from rastervision2.examples.sample_pipeline2.sample_pipeline2 import (
    SamplePipeline2Config)
from rastervision2.examples.deluxe_message_maker.deluxe_message_maker import (
    DeluxeMessageMakerConfig)


def get_config(runner, root_uri):
    names = ['alice', 'bob', 'susan']
    # Note that we use the DeluxeMessageMakerConfig and set the level to 3.
    message_maker = DeluxeMessageMakerConfig(greeting='hola', level=3)
    return SamplePipeline2Config(
        root_uri=root_uri, names=names, message_maker=message_maker)
