import math
from scipy.special import comb

# error parameters 
p = 1e-10
#q = 3

# toric code
d = 7
k = int(d/2) + 1
n = 2 * d**2

# probability logical error MWPM 
p_l_mwpm = 2 * 2 * d * comb(d, k) * ((2/3)*p)**k # 1/3 for only x errors, 2/3 for all three error types  
#print(p_l_mwpm, 'fail rate mwpm') 

# fail rate for only one operator
fail_rate_theory_1_operator = 2 * d * comb(d,k) / comb(2*d**2, k)
#print(fail_rate_theory_1_operator, 'fail rate theory alg and MWPM')
   
# probability logical error RL 
p_l_alg = d * (p/3)**k * ( 2 * comb(d,k) + k * comb(d, k-1))
#print(p_l_alg, 'fail rate alg')

# fail rate for three operators 
fail_rate_theory_3_operators = 4 *d * ( comb(d,k) + k * comb(d, k-1)) / (comb(2*d**2, k) * k**3)
print(fail_rate_theory_3_operators, 'fail rate 3 operators alg')
#print(1-fail_rate_theory_3_operators, 'success rate 3 operators alg')

number_of_ways = 0
for n_y in range(k):
    number_of_ways += comb(k,n_y) * comb(d, k-n_y) * comb(d-k+n_y, n_y)

fail_rate_mwpm_3_operators = (2*2*d*number_of_ways) / (comb(2*d**2, k) * k**3)
#print(fail_rate_mwpm_3_operators, 'fail mwpm')

#print(k, 'k')
#print(d, 'd')
#print(N_y,'N_y')



#print(comb(d,k-1) * comb(d-k+1, 1))
#print(comb(d,k-2) * comb(d-k+2, 2))
#print(comb(2,1))
############ check results

fail_rate_experimental = 1 - 9.997899999999999565e-01

asymptotic_fail = (fail_rate_experimental-fail_rate_theory_3_operators)/fail_rate_theory_3_operators * 100
#print(asymptotic_fail)

""" print( (2*p/3)**(k-1))
print( (p/3)**k) """



#print(p_conf, 'p_conf')
#p_conf = p**k * (1-p)**(n-k)
#print(p_k,'probability of k flips p_k')
#print(1/comb(10, 7)) 
#p_k = comb(n,k) * p_conf 

#print(5*4*3, 'number of combinations of 1 y and 2 x or z')
#print((0.1)**3)
#print((0.1*1/3) **1 * (0.1*1/3) **2)
#print((0.5*0.5) **1 * (0.5*0.5) **2)
#print(math.factorial(5)/ 4)
























