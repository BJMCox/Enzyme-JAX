import os
from .primitives import *


def default_nowheel_resource(dn):
    wheel_resource = os.path.join(dn, "..", "..", "clang", "staging")
    if os.path.isdir(os.path.join(wheel_resource, "include")):
        return wheel_resource
    return os.path.join(
        dn, "..", "..", "..", "external", "llvm-project", "clang", "staging"
    )


def default_linux_cflags():
    return ()
