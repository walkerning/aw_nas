# pylint: disable=invalid-name
import os
import re
import sys
import imp
import inspect
from collections import defaultdict

from aw_nas.utils.exception import PluginException
from aw_nas.utils.common_utils import get_awnas_dir
from aw_nas.utils import getLogger

LOGGER = getLogger("plugin")

plugins = []
plugin_modules = defaultdict(list)
import_errors = {}
norm_pattern = re.compile(r"[/|.]")

class AwnasPlugin(object):
    NAME = None

    dataset_list = []
    controller_list = []
    evaluator_list = []
    weights_manager_list = []
    objective_list = []
    trainer_list = []

    @classmethod
    def validate(cls):
        if not cls.NAME:
            raise PluginException("Your plugin needs a name.")

def is_valid_plugin(plugin_obj, existing_plugins):
    if inspect.isclass(plugin_obj) and issubclass(plugin_obj, AwnasPlugin) and \
       plugin_obj is not AwnasPlugin:
        plugin_obj.validate()
        return plugin_obj not in existing_plugins
    return False

def make_module(name, objects):
    #pylint: disable=protected-access
    LOGGER.debug('Creating module %s', name)
    name = name.lower()
    module = imp.new_module(name)
    module._PLUGIN_NAME = name.split('.')[-1]
    module._objects = objects
    module.__dict__.update((o.__name__, o) for o in objects)
    return module

def _reload_plugins():
    del plugins[:]
    plugin_modules.clear()
    plugin_dir = get_awnas_dir("AWNAS_PLUGIN_DIR", "plugins")
    LOGGER.info("Check plugins under %s", plugin_dir)

    # load plugins under directory (code from airflow)
    for root, _, files in os.walk(plugin_dir, followlinks=True):
        for f in sorted(files):
            try:
                filepath = os.path.join(root, f)
                if not os.path.isfile(filepath):
                    continue
                mod_name, file_ext = os.path.splitext(
                    os.path.split(filepath)[-1])
                if file_ext != ".py" or mod_name.startswith("test_"):
                    continue
                LOGGER.debug("Importing plugin module %s", filepath)
                # normalize root path as namespace
                # TODO: use the abspath or the relative path to `AWNAS_HOME/plugins`?
                # abspath CONs: others cannot easily use the dumped components
                # relpath CONs: cannot easily find the code that produce the results,
                #  since they all create the same namespace.
                #  Maybe this complexity/compatability should be handled by the plugin itself
                namespace = "_".join([re.sub(norm_pattern, "__", root), mod_name])

                m = imp.load_source(namespace, filepath)
                for obj in list(m.__dict__.values()):
                    if is_valid_plugin(obj, plugins):
                        plugins.append(obj)
                        export_plugin(obj)

            except Exception as e: #pylint: disable=broad-except
                LOGGER.exception(e)
                LOGGER.error("Failed to import plugin %s: %s", filepath, e)
                import_errors[filepath] = str(e)

    LOGGER.info("Loaded plugins: %s", ", ".join([p.NAME for p in plugins]))

def export_plugin(plugin):
    # for easy access: pop the components of each plugins under corresponding `
    # `aw_nas.<component name>.<plugin name>` module.
    # you can also diretly using RegistryMeta to access all components
    for compo_name in ["dataset", "controller",
                       "evaluator", "weights_manager", "objective", "trainer"]:
        base_mod = sys.modules["aw_nas." + compo_name]
        exported_compos = getattr(plugin, compo_name + "_list")
        if exported_compos:
            mod = make_module("aw_nas.{}.{}".format(compo_name, plugin.NAME), exported_compos)
            plugin_modules[compo_name].append(mod)
            sys.modules[mod.__name__] = mod
            setattr(base_mod, mod._PLUGIN_NAME, mod) #pylint: disable=protected-access
    return plugin_modules
