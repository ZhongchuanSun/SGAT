import os
import sys
import time
from configparser import ConfigParser
from collections import OrderedDict


class Configurer(object):
    def __init__(self):
        # get argument
        self.cmd_arg = self._load_cmd_arg()
        self.lib_arg = self._load_lib_arg()
        self.alg_arg = self._load_alg_arg()

        self._set_info()

    def _set_info(self):
        # parse the train file name to get dataset name
        train_file = self["train_file"]
        basename = os.path.basename(train_file)
        self.lib_arg["data_name"] = basename.split(".")[0]

    def _load_cmd_arg(self):
        cmd_arg = OrderedDict()
        for arg in sys.argv[1:]:
            arg_name, arg_value = arg.split("=")
            cmd_arg[arg_name[2:]] = arg_value

        return cmd_arg

    def _load_lib_arg(self):
        lib_file = "tfrec.ini"
        config = ConfigParser()
        config.optionxform = str
        config.read(lib_file, encoding="utf-8")
        lib_arg = OrderedDict(config["tfrec"].items())
        for arg in self.cmd_arg:
            if arg in lib_arg:
                lib_arg[arg] = self.cmd_arg[arg]

        return lib_arg

    def _load_alg_arg(self):
        alg_file = os.path.join("./conf", self["model"] + ".ini")
        config = ConfigParser()
        config.optionxform = str
        config.read(alg_file, encoding="utf-8")
        alg_arg = OrderedDict(config["hyperparameters"].items())
        for arg in self.cmd_arg:
            if arg in alg_arg:
                alg_arg[arg] = self.cmd_arg[arg]

        return alg_arg

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("index must be a str")

        if item in self.lib_arg:
            param = self.lib_arg[item]
        elif item in self.alg_arg:
            param = self.alg_arg[item]
        else:
            raise NameError("There are not the parameter named '%s'" % item)

        # convert param from str to value, i.e. int, float or list etc.
        try:
            value = eval(param)
            if not isinstance(value, (str, int, float, list, tuple, bool, None.__class__)):
                value = param
        except:
            if param.lower() == "true":
                value = True
            elif param.lower() == "false":
                value = False
            else:
                value = param

        return value

    def __contains__(self, o):
        return o in self.lib_arg or o in self.alg_arg

    def __str__(self):
        lib_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self.lib_arg.items()])
        alg_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self.alg_arg.items()])
        info = "\n\n%s's hyperparameters:\n%s\n\nTFRec hyperparameters:\n%s\n" % (self["model"], alg_info, lib_info)
        return info

    def __repr__(self):
        return self.__str__()


lib_config = Configurer()


def get_params_id():
    params_id = '_'.join(["{}={}".format(arg, value) for arg, value in lib_config.alg_arg.items() if len(value) < 20])
    special_char = {'/', '\\', '\"', ':', '*', '?', '<', '>', '|', '\t'}
    params_id = [c if c not in special_char else '_' for c in params_id]
    params_id = ''.join(params_id)
    params_id = params_id[:150]
    return params_id


_tfrec_run_id = None


def get_run_id():
    global _tfrec_run_id
    if _tfrec_run_id is None:
        params_id = get_params_id()

        data_name = lib_config["data_name"]
        model_name = lib_config["model"]
        timestamp = time.time()
        # data name, model name, param, timestamp
        _tfrec_run_id = "%s_%s_%s_%.8f" % (data_name, model_name, params_id, timestamp)

    return _tfrec_run_id
