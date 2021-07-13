class NasException(Exception):
    pass

class ConfigException(NasException):
    pass

class InvalidUseException(NasException):
    pass

class PluginException(NasException):
    pass

def expect(bool_expr, message="", exception_type=NasException):
    if not bool_expr:
        raise exception_type(message)
