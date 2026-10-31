"""Marketing Mix Model bayesiano sobre el demo de Robyn."""

import os
import warnings

# Sin compilador de C, PyTensor avisa en cada import. nutpie no lo necesita.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
warnings.filterwarnings("ignore", message="IProgress not found")
