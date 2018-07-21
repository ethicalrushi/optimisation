

import numpy as np
import math

#bechmark probability
pbench = np.array([0.1,0.3,0.9,0.3,0.8,0.94,0.86,0.82,0.41,0.28], dtype='f')  
n = 10 #no of persons


#premium for corresponding user
premium = np.array([1500,2134, 3129, 3000,4000, 1212, 1200,3000, 2111, 5000], dtype='f') 

#Random initialisation of incentives
inc = np.array([100,100,100,100,100,100,100,100,100,100], dtype='f')

inc_n = inc/400

rev = 0

eff = 10*(1-np.exp(-inc_n))

eff_n = eff/5
delta_p= (pbench)*(1- np.exp(-eff_n))/5

"""
for i in range(10):

	rev += (pbench[i]+delta_p[i])*premium[i] - inc[i]

"""
no_iter=100 #no of iterations
rate = 100.0 #learning rate

for i in range(no_iter):
	print(i)

	for i in range(10):

		rev += (pbench[i]+delta_p[i])*premium[i] - inc[i]

		#print('revenue:')
	print(rev)
	grad = pbench *(np.exp(-eff_n))/1000 *(np.exp(-inc_n))*premium -1
	inc = inc +rate* grad #grad ascend ---Concave optimisation----
	inc_n = inc/400
	eff = 10*(1-np.exp(-inc_n)) #updating effort

	eff_n = eff/5
	delta_p= (pbench)*(1- np.exp(-eff_n))/5 #updating increase in probability
	rev=0
	#print(grad)






"""
Just for testing

print('effort:')


for i in range(10):
	print(eff[i])



print('delta_p:')

for i in range(10):
	print(delta_p[i])
	



print('revenue:')
print(rev)

"""