import pandas as pd
import matplotlib.pyplot as plt
import numpy  as  np
import random
titanic_fname='C:/Users/liam/Desktop/deep learning/NLP/3/titanic/train.csv'
content=pd.read_csv(titanic_fname)
content=content.dropna()
age_with_fares=content[
(content['Age']>22)&(content['Fare']>130)&(content['Fare']<400)
]
sub_fare=age_with_fares['Fare']
sub_age=age_with_fares['Age']
plt.scatter(sub_age,sub_fare)
plt.show()#显示出图像

def func(age,k ,b):return k * age + b

def loss(y,yhat):
    # return np.mean(np.abs(y - yhat))
    # return np.mean(np.square(y - yhat))
    return np.mean(np.sqrt(y - yhat))
#

min_error_rate=float('inf')
best_k,best_b=None,None
loop_time=1000
losses=[]
change_directions=[
    #k b
    (+1,-1),#k increase b decrease
    (+1,+1),
    (-1,+1),
    (-1,-1)
]
k_hat = random.random() * 20 - 10  # kb取小数
b_hat = random.random() * 20 - 10
best_k, best_b = k_hat, b_hat
best_direction=None
def step():return random.random()*1
direction=random.choice(change_directions)

def derivate_k(y,yhat,x):
    abs_values=[1 if (y_i-yhat_i)>0 else -1 for y_i,yhat_i in zip(y,yhat)]
    return np.mean([a* -x_i for a , x_i in zip(abs_values,x)])

def derivate_b(y,yhat):
    abs_values=[1 if (y_i-yhat_i)>0 else -1 for y_i,yhat_i in zip(y,yhat)]
    return np.mean([a* -1 for a in abs_values])

learning_rate=1e-1
while loop_time>0:
    k_delta=-1*learning_rate*derivate_k(sub_fare,func(sub_age,k_hat,b_hat),sub_age)
    b_delta = -1*learning_rate*derivate_b(sub_fare, func(sub_age, k_hat, b_hat))
    k_hat+=k_delta
    b_hat+=b_delta
    # k_delta_direction,b_delta_direction=best_direction or random.choice(change_directions)
    # k_delta=k_delta_direction*step()
    # b_delta = b_delta_direction * step()
    # new_k=k_hat+k_delta
    # new_b = b_hat + b_delta
    # k_hat = random.randint(-10, 10)
    # b_hat = random.randint(-10, 10)
    # k_hat = random.random()*20-10#kb取小数
    # b_hat = random.random()*20-10
    estimated_fares=func(sub_age,k_hat,b_hat)
    error_rate=loss(y=sub_fare,yhat=estimated_fares)
    # print(error_rate)
    #加快收敛速度
    # if error_rate<min_error_rate:
    #     min_error_rate=error_rate
    #     best_k,best_b=k_hat,b_hat
    #     next_direction=(k_delta_direction,b_delta_direction)
    #     print(min_error_rate)
    print('loop=={}'.format(loop_time))
    # losses.append(min_error_rate)
    print('f(age)={}*age+{},with error rate:{}'.format(best_k,best_b,error_rate))
    # else:
    #     direction=random.choice(list(set(change_directions)-{(k_delta_direction,b_delta_direction)}))
    losses.append(error_rate)
    loop_time-=1
    # performance=loss(y=sub_fare,yhat=estimated_fares)
    # print(performance)

# plt.scatter(sub_age,sub_fare)
# plt.plot(sub_age,func(sub_age,best_k,best_b),c='r')
# plt.show()
plt.plot(range(len(losses)),losses)
plt.show()