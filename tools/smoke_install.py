"""Exercise an installed distribution and compare both inference backends."""
import numpy as np
import wade
from wade._wasserstein import HAVE_RUST_KERNEL
assert HAVE_RUST_KERNEL, "installed package has no current native kernel"
x=np.random.default_rng(43).poisson(20,(12,20));x[0,:3]*=20
cond=np.r_[np.ones(10,int),np.zeros(10,int)]
a=wade.wade(x,cond,max_permutations=79,seed=7,backend="rust")
b=wade.wade(x,cond,max_permutations=79,seed=7,backend="numpy")
for name in ("area_up","area_down","pvalue_up","pvalue_down","padj_up","padj_down","bulk_shift","upper_tail",
             "lower_tail","n_permutations","n_exceedances"):
    np.testing.assert_allclose(getattr(a,name),getattr(b,name),rtol=1e-13,atol=1e-13)
assert wade.manifest(a)["run"]["max_permutations"]==79
assert not hasattr(a,"p_subset")
print(wade.__version__,wade.__file__,"installed computation and NumPy parity passed")
