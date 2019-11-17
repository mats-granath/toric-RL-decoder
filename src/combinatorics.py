import math
from scipy.special import comb

# error parameters 
p = 1e-3
#q = 3

# toric code
d = 7
k = int(d/2) + 1
print(k, 'k')
n = 2 * d**2

# every possible combination of x,y and z in a line
number_of_ways = 0 
for n_y in range(k):
    number_of_ways += comb(d, n_y) * comb(d-n_y, k - n_y)


# probability logical error MWPM 
p_l_mwpm = 2 * 2 * d * number_of_ways * ((1/3)*p)**k
print(p_l_mwpm, 'p_l_mwpm') 

# fail rate for only one operator
#fail_rate_theory_1_operator = 2 * d * comb(d,k) / comb(2*d**2, k)
#print(fail_rate_theory_1_operator, 'fail rate theory alg and MWPM')
   
# probability logical error RL 
p_l_alg = 4 * d * ((1/3)*p)**k * (comb(d,k) + d * comb(d-1, k-1))
print(p_l_alg, 'pl_alg')

# fail rate for three operators 
fail_rate_theory_3_operators = 4*d * (comb(d,k) + d * comb(d-1, k-1)) / (comb(2*d**2, k) * k**3)
print(fail_rate_theory_3_operators, 'fail_rate_theory_3_operators')
#print(1-fail_rate_theory_3_operators, 'success rate 3 operators alg')

fail_rate_mwpm_3_operators = (2*2*d*number_of_ways) / (comb(2*d**2, k) * k**3)
print(fail_rate_mwpm_3_operators, 'fail_rate_mwpm_3_operators')


