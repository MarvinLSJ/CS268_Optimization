import numpy as np
import warnings

def GS(f, xs=0, s=1, mu=0, sigma=2):
    # optimization method used = Integrated Bracketing and Golden Section Algorithm


    # 1 Initialize randomly
    x1 = xs + np.random.normal(mu, sigma, 1)[0]
    tau = 0.618

    ep_R = 1e-7
    ep_abs = 1e-6

    # 2 Evaluate f2
    x2 = x1 + s

    # 3 Check and invert axis
    if f(x2) < f(x1):
        f2 = f(x1 + s)
    else:
        temp = x1
        x1 = x2
        x2 = temp
        s = -s
        f2 = f(x1 + s)

    # 4 Evaluate f4 based on tau

    x2s = []
    f2s = []
    x4s = []
    f4s = []

    s = s / tau
    x4 = x2 + s
    f4 = f(x4)

    x4s.append(x4)
    f4s.append(f(x4))

    # 5 6 Check and make sure f4 > f2
    while f4 <= f2:
        x1 = x2
        x2 = x4
        s = s / tau
        x4 = x2 + s
        f4 = f(x4)
        x4s.append(x4)
        f4s.append(f(x4))
    i = 0

    x3s = []
    f3s = []

    # do
    x3 = tau * x4 + (1 - tau) * x1
    f3 = f(x3)
    x3s.append(x3)
    f3s.append(f3)

    f_new = (f(x1) + f(x2) + f(x3)) / 3
    f_old = f_new + 1

    while (abs(x1 - x3) > ep_R * abs(x2) + ep_abs) and (abs(f_new - f_old) > ep_R * f(x2) + ep_abs):

        # 8 Loop on
        if (f2 < f3):
            x4 = x1
            x1 = x3
        else:
            x1 = x2
            x2 = x3
            f2 = f3

        # 7 Evaluate f3
        x3 = tau * x4 + (1 - tau) * x1
        f3 = f(x3)
        x3s.append(x3)
        f3s.append(f3)
        f_old = f_new
        f_new = (f(x1) + f(x2) + f(x3)) / 3
        i += 1

    return x2

def GS_mean(f, eva_count = 100):
    arr = []
    for i in range(eva_count):
        arr.append(GS(f))
    return np.array(arr).mean()

def normalize(v):
    dim = len(v)
    norm = np.linalg.norm(v)
    return [v[i]/norm for i in range(dim)]

def conjugate_gradients(func, x0, fprime,  CG_iter=2, verbose=0, avg_linesearch=0, x_tol=0.0005, f_tol=0.001, ep_R=1e-7):
# student code goes here:

    warnings.filterwarnings('error', 'overflow')
    xk = x0
    x_step = []
    x_step.append(xk)

    dim = len(xk)
    gk = [fprime[i](xk) for i in range(dim)]
    dk = normalize([-gk[i] for i in range(dim)])

    #     print('xk', xk)
    #     print('dk', dk)

    iter_count = 0
    f_old = func(xk)
    f_new = f_old + 1

    try:
        while (abs(f_new - f_old) > f_tol):  # ep_R*abs(f_old) + ep_abs):

            f_old = func(xk)

            for j in range(CG_iter):

                f_a = lambda alpha: func([xk[i] + alpha * dk[i] for i in range(dim)])
                # argmin alpha by GS line search
                if avg_linesearch == 1:
                    alpha_k = GS_mean(f_a)
                else:
                    alpha_k = GS(f_a)
                # print('\nalpha:%.9f' % alpha_k)
                xk1 = xk + np.dot(alpha_k, dk)
                #             print('xk1', xk1)
                #             print('f(xk1)', func(xk1))
                gk1 = [fprime[i](xk1) for i in range(dim)]
                beta_k = np.dot(gk1, gk1) / np.dot(gk, gk)
                #             print('beta:%.9f' % beta_k)

                dk1 = normalize([-gk1[i] for i in range(dim)] + np.dot(beta_k, dk))
                #             print('dk1', dk1)

                dk = dk1
                x_step.append(xk1)
                xk = xk1
                gk = gk1

            f_new = func(xk1)
            x_final = xk

            if (f_new >= 5 * f_old):
                if (verbose == 1):
                    print("\nEarly Stopping!")
                if (len(x_step) > 2):
                    x_final = x_step[len(x_step) - 2]
                break

            iter_count += 1
    except Warning:
        print('#### Overflow encountered!! ####')
        return

    if (verbose == 1):
        print('\nResult:', x_final, '%.6f' % func(x_final), iter_count, '\nSteps:', x_step)
    return x_final, func(x_final), iter_count

def alternate_method(func, x0, fprime,  CG_iter=2, verbose=0, avg_linesearch=0, x_tol=0.0005, f_tol=0.001, ep_R=1e-7):
    iter_count = 0
    xk = x0
    dim = len(xk)
    gk = [fprime[i](xk) for i in range(dim)]
    dk = normalize([-gk[i] for i in range(dim)])

    f_old = func(xk)
    f_new = f_old + 1

    while (abs(f_new - f_old) > f_tol):

        f_old = func(xk)

        f_a = lambda alpha: func([xk[i] + alpha * dk[i] for i in range(dim)])
        if avg_linesearch == 1:
            alpha_k = GS_mean(f_a)
        else:
            alpha_k = GS(f_a)

        xk1 = xk + np.dot(alpha_k, dk)
        f_new = func(xk1)

        xk = xk1
        iter_count += 1

    x_final = xk

    if (verbose == 1):
        print('\nResult:', x_final, '%.6f' % func(x_final), iter_count)
    return x_final, func(x_final), iter_count


"""
Input:
    func: Function need to be optimized
    init_pt: Initial point in n dimension
    fprime: Derivative function
    optimizer: choice from [steepest descent, conjugate gradient]
    x_opt: mathmatical optimum input
    f_opt: mathmatical optimum value

    eva_count: evaluate n times. Default: 100
    f_tol: function value tolerance, used in stop criterion. Default: 0.001
    avg_linesearch: set 1 to use average of 100 Golden Section line search results. Default: 0
    verbose: set 1 to print out variables for debugging. Default: 0
    plot: set 1 to plot estimated distribution and box plot of X error, function value error, time consumed.

Output:
    x_errs: List of errors between optimum and estimated X for every eva_count samples
    f_errs: List of errors between optimum and estimated function values for every eva_count samples
    iters: List of iteration number for every eva_count samples
    times: List of times in milliseconds for every eva_count samples

"""


def evaluate(func, init_pt, fprime, optimizer, x_opt, f_opt, CG_iter=2, eva_count=30, f_tol=0.001, avg_linesearch=0,
             verbose=0, plot=0):
    dim = len(x_opt)
    fs = []
    xs = []
    x_errs = []
    f_errs = []
    iters = []
    times = []
    for i in range(eva_count):
        #         init_pt_r = init_pt + np.random.normal(0, 5, 1)[0]
        start = time.time()
        x, f, iter_count = optimizer(func, init_pt, fprime, CG_iter, verbose, avg_linesearch, f_tol=f_tol)
        end = time.time()
        x_err = np.linalg.norm(x - x_opt)
        f_err = np.linalg.norm(f - f_opt)
        times.append(abs(start - end) * 1000)
        xs.append(x)
        fs.append(f)
        x_errs.append(x_err)
        f_errs.append(f_err)
        iters.append(iter_count)

    if plot == 1:
        fig = plt.figure(figsize=(12, 20))
        fig.suptitle('Scores', fontsize=20)

        plt.subplot(4, 2, 1)
        u = np.mean(x_errs)
        sig = np.mean(x_errs)
        x = np.linspace(u - 3 * sig, u + 3 * sig, 100)
        plt.title('Errors Normal Distribution')
        plt.plot(x, mlab.normpdf(x, u, sig))

        plt.subplot(4, 2, 2)
        plt.boxplot(x_errs, notch=True)
        plt.title('Errors Box Plot')

        plt.subplot(4, 2, 3)
        u = np.mean(fs)
        sig = np.mean(fs)
        x = np.linspace(u - 3 * sig, u + 3 * sig, 100)
        plt.title('Function Values Normal Distribution')
        plt.plot(x, mlab.normpdf(x, u, sig))

        plt.subplot(4, 2, 4)
        plt.boxplot(fs, notch=True)
        plt.title('Function Values Box Plot')

        plt.subplot(4, 2, 5)
        u = np.mean(times)
        sig = np.mean(times)
        x = np.linspace(u - 3 * sig, u + 3 * sig, 100)
        plt.title('Times Normal Distribution')
        plt.plot(x, mlab.normpdf(x, u, sig))

        plt.subplot(4, 2, 6)
        plt.boxplot(times, notch=True)
        plt.title('Times Box Plot')

        plt.show()

    return x_errs, f_errs, iters, times


if __name__=='__main__':
    test_func = lambda x: x[0]**2.0 + 15.0*x[0]*x[1] + 3.0*(x[1]**2.0)
    test_fprime = [lambda x: 2.0*x[0] + 15.0*x[1], lambda x: 6.0*x[1] + 15.0*x[0]]
    init_pt = (0.3, 0.1)
    print(conjugate_gradients(test_func, init_pt, test_fprime, 1))
    print(alternate_method(test_func, init_pt, test_fprime))


# Test for CG
    # x0 = [1, 3]
    # fprime = [lambda x: 2.0 * x[0], lambda x: 8.0 * x[1]]
    # func = lambda x: x[0] ** 2.0 + 4 * x[1] ** 2.0
    #
    # X_Err, F_Err, Iter, Time = evaluate(func, x0, fprime, conjugate_gradients, (0, 0), 0)
    # CG_res = pd.DataFrame([X_Err, F_Err, Iter, Time]).T
    # CG_res.columns = ['X_Err', 'F_Err', 'Iter', 'Time']
