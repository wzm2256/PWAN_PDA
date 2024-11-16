import pdb

from ..imagelist import ImageList
from ..office31 import Office31
from ..officehome import OfficeHome
from ..visda2017 import VisDA2017
from ..officecaltech import OfficeCaltech
from .imagenet_caltech import ImageNetCaltech
from .caltech_imagenet import CaltechImageNet
from ..domainnet import DomainNet
# from common.vision.datasets.partial.imagenet_caltech import ImageNetCaltech
from typing import Sequence, ClassVar


__all__ = ['Office31', 'OfficeHome', "VisDA2017", "CaltechImageNet", "ImageNetCaltech", "DomainNet"]


def partial(dataset_class: ClassVar, partial_classes: Sequence[str]) -> ClassVar:
    """
    Convert a dataset into its partial version.

    In other words, those samples which doesn't belong to `partial_classes` will be discarded.
    Yet `partial` will not change the label space of `dataset_class`.

    Args:
        dataset_class (class): Dataset class. Only subclass of ``ImageList`` can be partial.
        partial_classes (sequence[str]): A sequence of which categories need to be kept in the partial dataset.\
            Each element of `partial_classes` must belong to the `classes` list of `dataset_class`.

    Examples::

    >>> partial_classes = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard']
    >>> # create a partial dataset class
    >>> PartialOffice31 = partial(Office31, partial_classes)
    >>> # create an instance of the partial dataset
    >>> dataset = PartialDataset(root="data/office31", task="A")

    """
    if not (issubclass(dataset_class, ImageList)):
        raise Exception("Only subclass of ImageList can be partial")

    class PartialDataset(dataset_class):
        def __init__(self, **kwargs):
            super(PartialDataset, self).__init__(**kwargs)
            assert all([c in self.classes for c in partial_classes])
            samples = []
            for (path, label) in self.samples:
                class_name = self.classes[label]
                # print('{}_{}_{}'.format(path, label, (class_name in partial_classes)))
                if class_name in partial_classes:
                    samples.append((path, label))
            self.samples = samples
            self.partial_classes = partial_classes
            self.partial_classes_idx = [self.class_to_idx[c] for c in partial_classes]

    return PartialDataset


def default_partial(dataset_class: ClassVar, keep_num=0) -> ClassVar:
    """
    Default partial used in some paper.

    Args:
        dataset_class (class): Dataset class. Currently, dataset_class must be one of
            :class:`~common.vision.datasets.office31.Office31`, :class:`~common.vision.datasets.officehome.OfficeHome`,
            :class:`~common.vision.datasets.visda2017.VisDA2017`,
            :class:`~common.vision.datasets.partial.imagenet_caltech.ImageNetCaltech`
            and :class:`~common.vision.datasets.partial.caltech_imagenet.CaltechImageNet`.
    """
    if dataset_class == Office31:
        kept_classes = OfficeCaltech.CLASSES
    elif dataset_class == OfficeHome:
        if keep_num == 0:
            kept_classes = sorted(OfficeHome.CLASSES)[:25]
        else:
            kept_classes = sorted(OfficeHome.CLASSES)[:keep_num]
    elif dataset_class == VisDA2017:
        if keep_num == 0:
            kept_classes = sorted(VisDA2017.CLASSES)[:6]
        else:
            kept_classes = sorted(VisDA2017.CLASSES)[:keep_num]
    elif dataset_class == DomainNet:
        if keep_num == 0:
            kept_classes = sorted(DomainNet.CLASSES)[:40]
        else:
            kept_classes = sorted(DomainNet.CLASSES)[:keep_num]
    elif dataset_class in [ImageNetCaltech, CaltechImageNet]:
        kept_classes = dataset_class.CLASSES
    else:
        raise NotImplementedError("Unknown partial domain adaptation dataset: {}".format(dataset_class.__name__))
    # print(kept_classes)
    return partial(dataset_class, kept_classes)