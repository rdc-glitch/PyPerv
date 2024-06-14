import autograd.numpy as np
from autograd import grad, jacobian
from scipy.integrate import  odeint
import matplotlib.pylab as plt

plt.rcParams["figure.autolayout"] = True

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def generate_perversion(Gamma = 2/3, Lambda = 1, kappainit= 0.5):
    K = 1
    tau_0 = 0
#    Gamma = float(input('Value of Gamma :'))
    epsilon = 1e-4
    tres =0.02
#    Lambda = float(input('Value of Lambda :'))
    tailletheta =100

#    kappainit = float(input('value of kappa'))
    def kircchoff_equation(t, z):
    	F_1, F_2, F_3, kappa_1, kappa_2, kappa_3 = z
    	return [F_2*kappa_3-F_3*kappa_2, F_3*kappa_1-F_1*kappa_3, F_1*kappa_2-F_2*kappa_1, F_2+(Lambda-Gamma)*kappa_3*kappa_2, -(F_1+(1-Gamma)*kappa_3*kappa_1-K*kappa_3)/Lambda, ((1-Lambda)*kappa_1*kappa_2-K*kappa_2)/Gamma]

    def kircchoff_equation_backward(t, z):
    	F_1, F_2, F_3, kappa_1, kappa_2, kappa_3 = z
    	return [-(F_2*kappa_3-F_3*kappa_2), -(F_3*kappa_1-F_1*kappa_3), -(F_1*kappa_2-F_2*kappa_1), -(F_2+(Lambda-Gamma)*kappa_3*kappa_2), (F_1+(1-Gamma)*kappa_3*kappa_1-K*kappa_3)/Lambda, -((1-Lambda)*kappa_1*kappa_2-K*kappa_2)/Gamma]

    def kircchoff_equation_for_jacobian(z):
    	return np.array([z[1]*z[5]-z[2]*z[4], z[2]*z[3]-z[0]*z[5], z[0]*z[4]-z[1]*z[3], z[1]+(Lambda-Gamma)*z[5]*z[4], -(z[0]+(1-Gamma)*z[5]*z[3]-K*z[5])/Lambda, ((1-Lambda)*z[3]*z[4]-K*z[4])/Gamma])




    jacobianfunc = jacobian(kircchoff_equation_for_jacobian)

    def highest_curvature_point(x, y):
        # Ensure the input arrays are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Compute first derivatives using central differences
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Compute second derivatives using central differences
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Compute the curvature at each point
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**1.5
        curvature = numerator / denominator
        
        # Find the index of the maximum curvature
        max_curvature_index = np.argmax(curvature)
        
        # Return the point of highest curvature
        return max_curvature_index


    def donnesolutionpourthetatab(thetasize, theta0, theta1, T,  gamma, tau, kappa, vpunstfunc, vmstfunc, Xplusfunc, Xmoinsfunc):
    	thetalist= np.linspace(theta0, theta1, thetasize)	
    	t = np.arange(0.0,T, tres)
    	sizet = np.size(t)
    	solaroundunst = np.zeros((thetasize, sizet, 6))
    	i =-1
    	for thetaplus in thetalist:
    		i +=1
    		X_0 = Xplusfunc +epsilon*(np.real(vpunstfunc)*np.cos(thetaplus)+np.imag(vpunstfunc)*np.sin(thetaplus))
    		result_odeint = odeint(kircchoff_equation, X_0, t,  tfirst=True)
    		solaroundunst[i,:,0] = result_odeint[:,0]
    		solaroundunst[i,:,1] = result_odeint[:,1]
    		solaroundunst[i,:,2] = result_odeint[:,2]
    		solaroundunst[i,:,3] = result_odeint[:,3]
    		solaroundunst[i,:,4] = result_odeint[:,4]
    		solaroundunst[i,:,5] = result_odeint[:,5]
    	solaroundst = np.zeros((thetasize, sizet, 6))
    	i =-1
    	for thetamoins in thetalist:
    		i +=1
    		X_0 = Xmoinsfunc +epsilon*(np.real(vmstfunc)*np.cos(thetamoins)+np.imag(vmstfunc)*np.sin(thetamoins))
    		result_odeint = odeint(kircchoff_equation_backward, X_0, t,  tfirst=True)
    		solaroundst[i,:,0] = result_odeint[:,0]
    		solaroundst[i,:,1] = result_odeint[:,1]
    		solaroundst[i,:,2] = result_odeint[:,2]
    		solaroundst[i,:,3] = result_odeint[:,3]
    		solaroundst[i,:,4] = result_odeint[:,4]
    		solaroundst[i,:,5] = result_odeint[:,5]
    	return solaroundst, solaroundunst


    def plot_sol_shhoting(kappatest,tautest,  T1, T2):
        global  kappa_1p
        gammatest = K/kappatest-1+Gamma*(1-tau_0/tautest)
        kappatestm = kappatest
        tautestm = -tautest
        gammatestm = K/kappatestm-1+Gamma*(1-tau_0/tautestm)
        Xplus = np.array([gammatest*tautest*kappatest, 0.0, gammatest*tautest**2, kappatest, 0.0 , tautest])
        Jac = jacobianfunc(Xplus)
        eigenvalues = np.linalg.eig(Jac)
        #print(eigenvalues)
        a = 0
        for j in range(len(eigenvalues[0])):
            eigenvalue = eigenvalues[0][j]
            if np.real(eigenvalue) > 0.0001 and a ==0:
                vpunst = np.array(eigenvalues[1][j])
                a = 1
        Xmoins = np.array([gammatestm*tautestm*kappatestm, 0.0000, gammatestm*tautestm**2, kappatestm, 0.00 , tautestm])

        Jac = jacobianfunc(Xmoins)
        eigenvalues = np.linalg.eig(Jac)
        a = 0	
        for j in range(len(eigenvalues[0])):
            eigenvalue = eigenvalues[0][j]
            if np.real(eigenvalue) < -0.0001 and a ==0:
                vmst = np.array(eigenvalues[1][j])
                a = 1
        solaroundstestup, solaroundunstestup = donnesolutionpourthetatab(tailletheta, -np.pi, np.pi, T1,  gammatest, tautest, kappatest, vpunst, vmst, Xplus, Xmoins)
        solaroundstestdown, solaroundunstestdown = donnesolutionpourthetatab(tailletheta, -np.pi, np.pi, T2,  gammatestm, tautestm, kappatestm, vpunst, vmst, Xplus, Xmoins)
        return solaroundunstestup, solaroundstestdown


    tauinit = np.sqrt(1/Gamma*(-(kappainit-1/2)**2+1/4))


    deltalist = []
    def three_point_test():
        ## find approximate first perversion solution for each theta
        solaroundunstestup, solperv = plot_sol_shhoting(kappainit,tauinit, 1,300)
        score = []
        for j_theta in range(np.size(solperv[:,0,0])):
            sol_at_this_phase = solperv[j_theta,:,:]
            j_s = 0
            while (sol_at_this_phase[j_s, 5])*(sol_at_this_phase[j_s+1, 5]) > 0 :
                j_s +=1
            j_s +=1

            approx_perv_ind_v1 = j_s
            approx_perv_size = int(1/tauinit/tres)
            while (sol_at_this_phase[j_s, 5])*(sol_at_this_phase[j_s+1, 5]) >0:
                j_s +=1
            approx_perv_ind = approx_perv_ind_v1
            approx_psecond_perv_ind = j_s
            indtest_1 = approx_perv_ind
            indtest_2 = approx_perv_ind+10
            indtest_3 = approx_perv_ind+20
            try:
                norm = (sol_at_this_phase[indtest_3+approx_perv_size:approx_psecond_perv_ind, 0] - sol_at_this_phase[indtest_3, 0] )**2 
                for j in range(1, 6):
                    if j == 1 or j == 4:
                        norm +=  (sol_at_this_phase[indtest_3+approx_perv_size:approx_psecond_perv_ind, j] + sol_at_this_phase[indtest_3, j] )**2 
                    else:
                        norm +=  (sol_at_this_phase[indtest_3+approx_perv_size:approx_psecond_perv_ind, j] - sol_at_this_phase[indtest_3, j] )**2 
                       
                score3 = np.min(np.sqrt(  norm    ))

                norm = (sol_at_this_phase[indtest_2+approx_perv_size:approx_psecond_perv_ind, 0] - sol_at_this_phase[indtest_2, 0] )**2 
                for j in range(1, 6):
                    if j == 1 or j == 4:
                        norm +=  (sol_at_this_phase[indtest_2+approx_perv_size:approx_psecond_perv_ind, j] + sol_at_this_phase[indtest_2, j] )**2 
                    else:
                        norm +=  (sol_at_this_phase[indtest_2+approx_perv_size:approx_psecond_perv_ind, j] - sol_at_this_phase[indtest_2, j] )**2 
                score2 = np.min(np.sqrt(  norm    ))
                norm = (sol_at_this_phase[indtest_1+approx_perv_size:approx_psecond_perv_ind, 0] - sol_at_this_phase[indtest_1, 0] )**2 
                for j in range(1, 6):
                    if j == 1 or j == 4:
                        norm +=  (sol_at_this_phase[indtest_1+approx_perv_size:approx_psecond_perv_ind, j] + sol_at_this_phase[indtest_1, j] )**2 
                    else:
                        norm +=  (sol_at_this_phase[indtest_1+approx_perv_size:approx_psecond_perv_ind, j] - sol_at_this_phase[indtest_1, j] )**2 
                score1 = np.min(np.sqrt(  norm    ))
            except:
                score1 = 1e8
                score2 = 1e8
                score3 = 1e8
            score.append(np.mean([score1, score2, score3]))
        meanscore = np.mean(score)
        minscore = np.min(score)
        lenhomo = 0
        j_thetalistkept = []
        seuil = 2
        for j_theta in range(np.size(solperv[:,0,0])):
            if score[j_theta] < minscore+(meanscore-minscore)/seuil:
                lenhomo += 1
                j_thetalistkept.append(j_theta)
        homosol = np.zeros((lenhomo,np.size(sol_at_this_phase[:, 5]), 6 ))
        j_homo = 0
        for j_theta in range(np.size(solperv[:,0,0])):
            sol_at_this_phase = solperv[j_theta,:,:]
            if score[j_theta] < minscore+(meanscore-minscore)/seuil:
                homosol[j_homo, : , : ]  = sol_at_this_phase[:,:]
                j_homo += 1
        print(lenhomo)
        taillelambda = 4000
        indcutlist = []
        for j in range(lenhomo):
            sol_at_this_phase = solperv[j_thetalistkept[j],:,:]
            #plt.show()
            j_s = 0
            while (sol_at_this_phase[j_s, 5])*(sol_at_this_phase[j_s+1, 5]) > 0 :
                j_s +=1
            approx_perv_ind_v1 = j_s
            approx_perv_size = int(1/tauinit/tres)
            j_s +=1
            while (sol_at_this_phase[j_s, 5])*(sol_at_this_phase[j_s+1, 5]) >0:
                j_s +=1
            approx_perv_ind = approx_perv_ind_v1
            approx_psecond_perv_ind = j_s
            indcut = highest_curvature_point(np.sqrt(sol_at_this_phase[approx_perv_ind:approx_psecond_perv_ind, 3]**2+sol_at_this_phase[approx_perv_ind:approx_psecond_perv_ind, 4]**2), sol_at_this_phase[approx_perv_ind:approx_psecond_perv_ind, 5])
            sol_test = sol_at_this_phase[:approx_perv_ind+indcut, :]
            sol_test = sol_test[:, 3:]

            kap = sol_test[0,0]
            taup = sol_test[0,2]
            sol_test = np.flip(sol_test[:,:], axis = 0)

            solextended = np.zeros((taillelambda, 3))
            solextended[:np.size(sol_test[:,0]),:] = sol_test[:, :]
            solextended[np.size(sol_test[:,0]):taillelambda, 0] = kap
            solextended[np.size(sol_test[:,0]):taillelambda, 2] = taup
            sol_test = solextended
            sol_test = np.flip(sol_test[:,:], axis = 0)

            indcutlist.append(approx_perv_ind+indcut)
            deltalist.append( (sol_at_this_phase[approx_perv_ind+indcut, 3]-kappainit)**2+ (sol_at_this_phase[approx_perv_ind+indcut, 5]-tauinit)**2)

        if len(deltalist) < 1:
            return 0
        else:
            i = np.argmin(deltalist)
            return  (solperv[j_thetalistkept[i],:indcutlist[i],:])

    while len(deltalist) < 1:
        solperv_homo_long = three_point_test()
        tailletheta = tailletheta*2



    for i in range(np.size(solperv_homo_long[:,5])):
        indperv = np.argmin((solperv_homo_long[:,5]**2))
    solcomplete = np.zeros(( 2*indperv, 3))
    solcomplete[:indperv, :] = solperv_homo_long[:indperv,3:]
    solcomplete[indperv:, :] = np.flip(solperv_homo_long[:indperv,3:], axis = 0)
    solcomplete[indperv:, 2] = -solcomplete[indperv:, 2]
    solperv_homo_long = solcomplete
    plt.plot(np.sqrt(solperv_homo_long[:,0]**2+solperv_homo_long[:,1]**2), solperv_homo_long[:,2])

    plt.show()




generate_perversion()


