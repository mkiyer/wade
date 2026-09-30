"""Exercise an installed distribution: both tests run, repeat exactly under a seed, and keep their records."""
import numpy as np
import wade
x=np.random.default_rng(43).poisson(20,(12,20));x[0,:3]*=20
cond=np.r_[np.ones(10,int),np.zeros(10,int)]
run=lambda: wade.wade(x,cond,max_permutations=79,seed=7)
a,b=run(),run()
for name in wade.RESULT_COLUMNS[2:]+("mean_permutations","mean_exceedances","subset_permutations","subset_exceedances"):
    np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
assert a.subset(0).side=="up" and wade.manifest(a)["run"]["max_permutations"]==79
print(wade.__version__,wade.__file__,"installed computation passed")
