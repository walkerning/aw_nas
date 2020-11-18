# -*- coding: utf-8 -*-
"""A simple registry meta class.
"""

import abc
import collections

from aw_nas.utils import getLogger

__all__ = ["RegistryMeta", "RegistryError"]

LOGGER = getLogger("registry")

class RegistryError(Exception):
    pass

def _default_dct_of_list():
    return collections.defaultdict(list)

class RegistryMeta(abc.ABCMeta):
    registry_dct = collections.defaultdict(dict)
    supported_rollout_dct = collections.defaultdict(_default_dct_of_list)

    def __init__(cls, name, bases, namespace):
        super(RegistryMeta, cls).__init__(name, bases, namespace)
        ## DEPRECATED: the interface of every class is defined explicitly in the arguments
        ##             instead of a cover-all config dict,
        ##             as failing loudly can avoid subtle bugs (e.g. mistyping)
        # # handle default_cfg
        # if "default_cfg" in attrs:
        #     cfg = six.moves.reduce(lambda d, new: d.update(new) or d,
        #          [bcls.default_cfg for bcls in reversed(bases) if hasattr(bcls, "default_cfg")],
        #                            {})
        #     cfg.update(cls.default_cfg)
        #     cls.default_cfg = cfg

        if hasattr(cls, "REGISTRY"):
            # register the class
            table = cls.REGISTRY
            abstract_methods = cls.__abstractmethods__
            if not abstract_methods:
                entry = namespace.get("NAME", name.lower())
                setattr(cls, "NAME", entry)
                RegistryMeta.registry_dct[table][entry] = cls
                LOGGER.debug("Register class `%s` as entry `%s` in table `%s`.",
                             name, entry, table)

                if cls.REGISTRY == "rollout":
                    # allow new defined rollout class to declare which component can be reused
                    if hasattr(cls, "supported_components"):
                        for registry, type_ in cls.supported_components:
                            RegistryMeta.supported_rollout_dct[registry][type_].append(entry)
            else:
                if "NAME" in namespace:
                    entry = namespace["NAME"]
                    LOGGER.warning("Can't register abstract class `%s` as entry `%s`"
                                   " in table `%s`, ignore. Abstract methods: %s",
                                   name, entry, table, ", ".join(abstract_methods))

    @classmethod
    def get_class(mcs, table, name):
        try:
            return mcs.all_classes(table)[name]
        except KeyError:
            raise RegistryError("No registry item {} available in registry {}."
                                .format(name, table))

    @classmethod
    def all_classes(mcs, table):
        try:
            return mcs.registry_dct[table]
        except KeyError:
            raise RegistryError("No registry table {} available.".format(table))

    @classmethod
    def avail_tables(mcs):
        return mcs.registry_dct.keys()

    def all_classes_(cls):
        return RegistryMeta.all_classes(cls.REGISTRY)

    def get_class_(cls, name):
        return RegistryMeta.get_class(cls.REGISTRY, name)

    def registered_supported_rollouts_(cls):
        return RegistryMeta.supported_rollout_dct[cls.REGISTRY][cls.NAME]
