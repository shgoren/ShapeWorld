from shapeworld.dataset import CaptionAgreementDataset, TextSelectionMultiShapeDataset
from shapeworld.generators import RandomAttributesGenerator
from shapeworld.captioners import RegularTypeCaptioner


class MultishapeSimpleTextselect(TextSelectionMultiShapeDataset):

    def __init__(
        self,
        entity_counts=(3,),
        train_entity_counts=(3,),
        validation_entity_counts=(3,),
        test_entity_counts=(3,),
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid'), ('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        test_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid'), ('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=6,
        vocabulary=('.', 'a', 'an', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'yellow'),
        language=None,
        number_texts=10
    ):

        world_generator = RandomAttributesGenerator(
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
            max_provoke_collision_rate=0.0,
            collision_tolerance=0.0,
            boundary_tolerance=0.0
        )
        world_generator.world_size = 128

        world_captioner = RegularTypeCaptioner()

        super(MultishapeSimpleTextselect, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            language=language,
            number_texts=number_texts
        )


dataset = MultishapeSimpleTextselect
