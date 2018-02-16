"""Base class used for defining network params.

Author: Shubham Toshniwal
Contact: shtoshni@ttic.edu
Date: February, 2018
"""

from bunch import Bunch

class BaseParams(object):
    """Base class for dealing with parameters."""

    @classmethod
    def class_params(cls):
        return Bunch()

    @classmethod
    def add_parse_options(cls):
        pass

    @classmethod
    def get_updated_params(cls, options):
        params = cls.class_params()
        for attr in params.keys():
            if attr in options:
                if type(params[attr]) == type(options[attr]):
                    params[attr] = options[attr]
        return params
