#hw4 - q1
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
desiredF = 0
GAMMA= 0.99
#%%


def PlotConvergenceOfF3d(x,y,f_all_values,titleName):
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-1,1,2/1000)
    Y = np.arange(-1,1,2/1000)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y)
    # Plot the surface.
    ax.plot_surface(X, Y, Z,linewidth=0, antialiased=False)
    ax.plot(x, y, f_all_values)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Convergence of f')
    plt.show(block=False)
    plt.savefig('q1-'+titleName+'_f_3d.png')
    plt.show(block=False)

def PlotConvergenceOfF(f_all_values,titleName):
    x = list(np.arange(1,len(f_all_values)+1,1))
    plt.figure()
    plt.plot(x,f_all_values)
    plt.xlabel('step')
    plt.ylabel('f')
    plt.title('Convergence of f')
    plt.show(block=False)
    plt.savefig('q1-'+titleName+'_f.png')
def PlotConvergence_X_Y(x_all_values,y_all_values,titleName):
    plt.figure()
    plt.plot(x_all_values,y_all_values)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y Vs. x')
    plt.show(block=False)
    plt.savefig('q1-'+titleName+'_x_y.png')

def func(x,y):
    f=80*(x**4)+0.01*(y**6)
    return f

def grad_f(x,y):
    gradX = 4*80*(x**3)
    gradY = 6*0.01*(y**5)
    return gradX,gradY

def Hessian(x,y):
    matHessian = np.array([[3*4*80*(x**2),0],[0,5*6*0.01*(y**4)]])
    return matHessian

def GetDelta(eta,x, y,alg,m_n_1):
    delX=1e-15
    delY=1e-15
    m_n=0
    if alg=='grad':
        gradX, gradY = grad_f(x, y)
        delX, delY = -eta * gradX, -eta * gradY
    elif alg=='neuton':
        delta = -np.linalg.inv(Hessian(x, y)) * np.array([grad_f(x, y)])
        delX, delY = delta[0,0], delta[1,1]
    elif alg=='momentum':
        gradX, gradY = grad_f(x, y)
        m_n_p1 = [GAMMA*mi for mi in m_n_1]
        m_n_p2 = [eta*gradX,eta*gradY]
        m_n = [m_n_p1[0]+m_n_p2[0],m_n_p1[1]+m_n_p2[1]]
        [delX, delY]= [-1*i for i in m_n]
    return delX, delY,m_n

def GD(x0,y0,eta,thresh,alg,titleName,MAX_ITERATIONS=5):
    i=0
    x_all_values = [x0]
    y_all_values = [y0]
    f_all_values = [func(x0,y0)]
    (x,y)=(x0,y0)
    m_n_1 = [0,0]
    for i in range(MAX_ITERATIONS):
        delX, delY,m= GetDelta(eta,x, y,alg,m_n_1)
        m_n_1 = m
        x_t_1,y_t_1 = x + delX, y + delY
        f = func(x_t_1, y_t_1)
        x_all_values.append(x_t_1)
        y_all_values.append(y_t_1)
        f_all_values.append(f)
        if np.abs(f-desiredF) <= thresh:
            break
        else:
            (x,y)=(x_t_1,y_t_1)
    if i==(MAX_ITERATIONS-1):
        print(r'The algorithm did not converage!')
        exit(-1)
    print('max steps is: '+str(i+1))
    PlotConvergenceOfF3d(x_all_values,y_all_values,f_all_values,titleName)
    PlotConvergenceOfF(f_all_values,titleName)
    PlotConvergence_X_Y(x_all_values,y_all_values,titleName)
    return func(x_t_1,y_t_1)

def q_b():
    eta = 0.01
    (x0, y0) = (1., 1.)
    thresh = 1e-4
    titleName = 'b'
    alg = 'grad'
    f = GD(x0, y0, eta, thresh,alg,titleName)
    print('final f is '+str(f))
    print('finish q1-b!')

def q_c():
    eta = 0.001
    (x0, y0) = (1., 1.)
    thresh = 1e-4
    titleName = 'c'
    alg = 'grad'
    f = GD(x0, y0, eta, thresh,alg,titleName,int(1e11))
    print('final f is ' + str(f))
    print('finish q1-c!')

def q_d():
    eta=1e10
    (x0, y0) = (1., 1.)
    thresh = 1e-4
    alg = 'neuton'
    titleName = 'd'
    f = GD(x0, y0, eta, thresh,alg,titleName,int(1e11))
    print('final f is ' + str(f))
    print('finish q1-d!')
def q_f():
    eta = 0.001
    (x0, y0) = (1., 1.)
    thresh = 1e-4
    alg = 'momentum'
    titleName = 'f'
    f = GD(x0, y0, eta, thresh,alg, titleName, int(1e11))
    print('final f is ' + str(f))
    print('finish q1-f!')
def main():
    q_b()
    q_c()
    q_d()
    q_f()
    print('finish main!')
if __name__=='__main__':
    main()
    print('finish progress!')
