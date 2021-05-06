import os
import sys
from importlib import resources


_locale = 'English_United States.1252' if os.name == 'nt' else 'en_US.UTF-8'

_CONFIG = {
    'ws': '',
    'locale': _locale,
    'msg': True,
    'refresh': None,
    'export': None,
}

_filename, _ = os.path.splitext(os.path.basename(sys.argv[0]))
_DBNAME = _filename + '.db'
_GRAPH_NAME = _filename + '.gv'
_JOBS = {}
# folder name (in workspace) for temporary databases for parallel work
_CONN = [None]
_TEMP = "_temp"

# sqlite3 keywords
with resources.open_text('mapon', 'sqlite_keywords.txt') as f:
    _RESERVED_KEYWORDS = set(x.strip() for x in f.readlines())

