from invoke import Collection

from .base_git_commands import close, gacp, issues, newbranch, newissue
from .setup_repo import labels
from . import buckets
from . import scenario


ns = Collection()
ns.add_task(gacp)
ns.add_task(close)
ns.add_task(issues)
ns.add_task(newbranch)
ns.add_task(labels)
ns.add_task(newissue)
ns.add_collection(Collection.from_module(buckets))
ns.add_collection(Collection.from_module(scenario))