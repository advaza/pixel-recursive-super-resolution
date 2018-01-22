# Copyright (c) 2017 Lightricks. All rights reserved.
"""
A singleton wrapper of ArgumentParser class.
"""

import sys, yaml, json, pickle

from argparse import ArgumentParser, Namespace


def try_pickle_parse(config_file_path):
    try:
        with open(config_file_path, 'rb') as config_file:
            config_file_args = pickle.load(config_file)
    except (ValueError, TypeError, pickle.UnpicklingError):
        return None
    return config_file_args


def try_json_parse(config_file_path):
    try:
        with open(config_file_path, 'r') as config_file:
            config_file_args = json.loads(config_file)
    except (ValueError, TypeError):
        return None
    return config_file_args


def try_yml_parse(config_file_path):
    try:
        with open(config_file_path, 'rb') as config_file:
                config_file_args = yaml.load(config_file)
    except (ValueError, TypeError):
        return None
    return config_file_args


class MyArguments(ArgumentParser):

    def __init__(self):

        self.fromfile_prefix_chars = ["@"]
        ArgumentParser.__init__(self, fromfile_prefix_chars=self.fromfile_prefix_chars)
        self.args = None
        self.config_files_data = {}

    def __getattr__(self, name):

        if self.args is None:
            self.parse_args()

        return self.args.__dict__[name]

    def _default_file_parse(self, config_file_path, new_arg_strings):

        with open(config_file_path) as config_file:
            try:
                arg_strings = []
                for arg_line in config_file.read().splitlines():
                    for arg in self.convert_arg_line_to_args(arg_line):
                        arg_strings.append(arg)
                arg_strings = self._read_args_from_files(arg_strings)
                new_arg_strings.extend(arg_strings)
            except OSError:
                err = sys.exc_info()[1]
                self.error(str(err))

    def _read_args_from_files(self, arg_strings):
        """
        Override function.
        Extend the parent function by adding more options
        for configuration files, other than arguments per line.
        :param arg_strings:
        :return:
        """
        # expand arguments referencing files
        new_arg_strings = []
        for arg_string in arg_strings:

            # for regular arguments, just add them back into the list
            if not arg_string or arg_string[0] not in self.fromfile_prefix_chars:
                new_arg_strings.append(arg_string)

            # replace arguments referencing files with the file content
            else:
                config_file_path = arg_string[1:]
                config_file_args = try_yml_parse(config_file_path=config_file_path)

                if not config_file_args:
                    config_file_args = try_json_parse(config_file_path=config_file_path)

                if config_file_args:
                    self.config_files_data.update(config_file_args)
                else:

                    self._default_file_parse(
                        config_file_path=config_file_path,
                        new_arg_strings=new_arg_strings
                    )

        # return the modified argument list
        return new_arg_strings

    def process_default_namespace(self, namespace):

        default_namespace_dict = {}
        if self.config_files_data:
            default_namespace_dict.update(self.config_files_data)
        if namespace:
            default_namespace_dict.update(namespace._get_kwarg)

        default_namespace = None
        if default_namespace_dict:
            default_namespace = Namespace(**default_namespace_dict)
        return default_namespace

    def parse_args(self, args=None, namespace=None):

        if args is None:
            args = sys.argv[1:]

        config_files_args = [arg for arg in args if arg[0] in self.fromfile_prefix_chars]
        other_args = [arg for arg in args if arg[0] not in self.fromfile_prefix_chars]

        self.args = ArgumentParser.parse_args(self, args=config_files_args)
        default_namespace = self.process_default_namespace(namespace)

        # Namespace arguments overwritten be command line arguments.
        self.args = ArgumentParser.parse_args(self, args=other_args, namespace=default_namespace)
        return self.args


class SingletonDecorator:

    def __init__(self, klass):
        self.klass = klass
        self.instance = None
        self.args = None

    def __call__(self, *args, **kwds):
        if self.instance is None:
            self.instance = self.klass(*args, **kwds)
        return self.instance


Flags = SingletonDecorator(MyArguments)
