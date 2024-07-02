import autograd.numpy as np
from autograd import grad, jacobian
from scipy.integrate import  odeint
import matplotlib.pylab as plt
from numpy import *

plt.rcParams["figure.autolayout"] = True

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
font = {'family': 'Latin Modern Roman',
    'weight': 'normal',
    'size': 20,
    }
def translation_perversion():
    K = 1
    tau_0 = 0
    Gamma = (input('Value of Gamma (positive float) - press enter for default value - :'))
    if Gamma == '':
        Gamma = 2/3
    else:
        Gamma = float(Gamma)
    epsilon = 1e-4
    tres =0.03
    Lambda = (input('Value of Lambda (positive float) - press enter for default value - :'))
    tailletheta =1000
    if Lambda == '':
        Lambda = 1
    else:
        Lambda = float(Lambda)
    numberofcoils = input('Number of coils of the rod in its natural state - press enter for default value -:')
    if numberofcoils == '':
        numberofcoils = 10
    else:
        numberofcoils = float(numberofcoils)

    resolution_force = (input('Number of points in force (positive integer) - press enter for default value 10 - :'))
    if resolution_force == '':
        resolution_force = 10
    else:
        resolution_force = int(resolution_force)

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
    sizedesiree = int(2*np.pi*numberofcoils/tres)
    Xbig = np.zeros((resolution_force, sizedesiree, 3))
    kappabig = np.zeros((resolution_force, sizedesiree, 3))
    Tafaire = np.linspace(0.15, 0.8/Gamma, resolution_force)
    figfin= plt.figure()
    axfin = figfin.add_subplot(111)
    for axialload in Tafaire:
        tailletheta = 500
        print(axialload)
        kappalist = np.linspace(0.01,0.99, 1000)
        taulist = np.sqrt(1/Gamma*(-(kappalist-1/2)**2+1/4))
        Tlist = (1/kappalist-1+Gamma)*taulist*np.sqrt(taulist**2+kappalist**2)
        indinit = np.argmin((Tlist-axialload)**2)
        kappainit = kappalist[indinit]
        tauinit = taulist[indinit]
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
                if np.sqrt(sol_at_this_phase[approx_perv_ind, 3]**2+sol_at_this_phase[approx_perv_ind, 4]**2) < kappainit:
                    deltalist.append( (sol_at_this_phase[approx_perv_ind+indcut, 3]-kappainit)**2+ (sol_at_this_phase[approx_perv_ind+indcut, 5]-tauinit)**2)
                else:
                    deltalist.append( 1e9)
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
        tailleact = np.size(solperv_homo_long[:,0])
        if np.size(solperv_homo_long[:,0]) > sizedesiree:
            solperv_homo_long = solperv_homo_long[int((np.size(solperv_homo_long[:,0])-sizedesiree)/2):-int((np.size(solperv_homo_long[:,0])-sizedesiree)/2),:]
        else:
            newsol = np.zeros((sizedesiree, 3))
            try:
                newsol[int(-(np.size(solperv_homo_long[:,0])-sizedesiree)/2):-int(-(np.size(solperv_homo_long[:,0])-sizedesiree)/2),:] = solperv_homo_long
                newsol[:int(-(np.size(solperv_homo_long[:,0])-sizedesiree)/2),:] = solperv_homo_long[0,:]
                newsol[-int(-(np.size(solperv_homo_long[:,0])-sizedesiree)/2):,:] = solperv_homo_long[-1,:]
            except:
                newsol[int(-(np.size(solperv_homo_long[:,0])-sizedesiree)/2):-int(-(np.size(solperv_homo_long[:,0])-sizedesiree)/2)-1,:] = solperv_homo_long
                newsol[:int(-(np.size(solperv_homo_long[:,0])-sizedesiree)/2),:] = solperv_homo_long[0,:]
                newsol[-int(-(np.size(solperv_homo_long[:,0])-sizedesiree)/2)-1:,:] = solperv_homo_long[-1,:]                
            solperv_homo_long = newsol
            #plt.plot(newsol)
            #plt.show()
        cm = 1/2.54
        fig = plt.figure(figsize=plt.figaspect(1.5 ))
        ax3 = fig.add_subplot(2, 1, 1)
        ax3.plot(solperv_homo_long[:,2], np.sqrt(solperv_homo_long[:,0]**2+solperv_homo_long[:,1]**2), 'r-', linewidth = 3)
        axfin.plot(solperv_homo_long[:,2], np.sqrt(solperv_homo_long[:,0]**2+solperv_homo_long[:,1]**2), 'k-')
        ax3.set_ylim((np.min(np.sqrt(solperv_homo_long[:,0]**2+solperv_homo_long[:,1]**2))*0.666, 1.333*np.max(np.sqrt(solperv_homo_long[:,0]**2+solperv_homo_long[:,1]**2))))
        ax3.tick_params( length=4, width=0.5)
        ax3.set_ylabel(r'$\tau/\kappa_0$', fontsize = 15)
        ax3.set_xlabel(r'$\kappa/\kappa_0$', fontsize = 15)
        for label in ax3.yaxis.get_majorticklabels():
            label.set_fontsize(5)
        for label in ax3.xaxis.get_majorticklabels():
            label.set_fontsize(5)
        #plt.show()
        manifold = solperv_homo_long
        size = np.size(manifold[:,2])
        #manifoldbis = np.zeros((2*np.size(manifold[:,2]), 3))
        #manifoldbis[:np.size(manifold[:,2]),:] = manifold
        #manifoldbis[np.size(manifold[:,2]):,0] = np.flip(manifold[:,0], axis = 0)
        #manifoldbis[np.size(manifold[:,2]):,1] = -np.flip(manifold[:,1], axis = 0)
        #manifoldbis[np.size(manifold[:,2]):,2] = np.flip(manifold[:,2], axis = 0)


        #manifold = manifoldbis
        size = np.size(manifold[:,2])
        theta0 = np.pi/2
        t = np.linspace(0, 2*np.pi, 100)

        xi = np.pi/2-np.arctan(manifold[:, 1]/manifold[:, 0])
        ds = tres
        #print(ds)
        xidev = np.zeros(size)
        for i in range(1,size):
            xidev[i] = (xi[i]-xi[i-1])/ds
        kappafren, taufren = np.sqrt(manifold[:, 0]**2+manifold[:, 1]**2), manifold[:, 2]+xidev
        #kappafren, taufren = manifold[:,0]*np.ones((size,1)), manifold[:,2]
        temps = np.linspace(0,1, size)
        T = np.zeros((size, 3))
        N = np.zeros((size, 3))
        B = np.zeros((size, 3))


        T[0,:],N[0,:], B[0,:] = [0, kappafren[0]/np.sqrt(kappafren[0]**2+taufren[0]**2),taufren[0]/np.sqrt(kappafren[0]**2+taufren[0]**2)], [1, 0, 0], [0, -taufren[0]/np.sqrt(kappafren[0]**2+taufren[0]**2), kappafren[0]/np.sqrt(kappafren[0]**2+taufren[0]**2)] #[0, 0,1], [1, 0, 0], [0,1,0] 
        for i in range(1, size):
            #print(np.sum(T[i-1, :]**2))
            T[i, :] = (T[i-1, :] + ds*kappafren[i-1] * N[i-1,:]  )/np.sum((T[i-1, :] + ds*kappafren[i-1] * N[i-1,:]  )**2)
            N[i, :] = (N[i-1, :] - ds*kappafren[i-1] * T[i-1,:]  + ds*taufren[i-1]*B[i-1,:])/np.sum((N[i-1, :] - ds*kappafren[i-1] * T[i-1,:]  + ds*taufren[i-1]*B[i-1,:])**2)
            B[i, :] = (B[i-1, :] - ds*taufren[i-1] * N[i-1,:]  )/np.sum((B[i-1, :] - ds*taufren[i-1] * N[i-1,:]  )**2)


        X_0 = [-manifold[0, 0]/(manifold[0, 0]**2+manifold[0, 2]**2)*0.5, manifold[0, 0]/(manifold[0, 0]**2+manifold[0, 2]**2)*np.sqrt(3)/2,0]
        X = np.zeros((np.size(temps), 3))
        X[0, :] = X_0
        for i in range(1,np.size(T[:,0])):
            X[i, :]= X[i-1,:]+ ds*T[i-1,:]


        size = np.size(manifold[:,2])
        xi = np.pi/2-np.arctan(manifold[:, 1]/manifold[:, 0])
        ds = 0.1
        #print(ds)
        xidev = np.zeros(size)
        for i in range(1,size):
            xidev[i] = (xi[i]-xi[i-1])/ds
        kappafren, taufren = np.sqrt(manifold[:, 0]**2+manifold[:, 1]**2), manifold[:, 2]-xidev

        temps = np.linspace(0,1, size)
        d3 = np.zeros((size, 3))
        d1 = np.zeros((size, 3))
        d2 = np.zeros((size, 3))


        d3[0,:],d2[0,:], d1[0,:]= [0, 0,1],  [0,1,0], [1, 0, 0] # = -np.array([0, kappafren[0]/np.sqrt(kappafren[0]**2+taufren[0]**2),taufren[0]/np.sqrt(kappafren[0]**2+taufren[0]**2)]), np.array([1, 0, 0]), np.array([0, -taufren[0]/np.sqrt(kappafren[0]**2+taufren[0]**2), kappafren[0]/np.sqrt(kappafren[0]**2+taufren[0]**2)]) 
        for i in range(1, size):
            #print(np.sum(T[i-1, :]**2))
            d3[i, :] = (d3[i-1, :] +ds*(manifold[i-1, 1] * d1[i-1,:]) -manifold[i-1, 0] * d2[i-1,:] )
            d2[i, :] = (d2[i-1, :] + ds*(manifold[i-1, 0] * d3[i-1,:]-manifold[i-1, 2] * d1[i-1,:])  )
            d1[i, :] = (d1[i-1, :] + ds*(manifold[i-1, 2] * d2[i-1,:]-manifold[i-1, 1] * d3[i-1,:])  )


        def generate_circle_by_vectors(t, C, r, n, u):
            n = n/np.linalg.norm(n)
            u = u/np.linalg.norm(u)
            P_circle = r*cos(t)[:,newaxis]*u + r*sin(t)[:,newaxis]*cross(n,u) + C
            return P_circle
        #fig = plt.figure()
        ax3d = fig.add_subplot(2, 1, 2, projection='3d')
        thetaefflist = [0,np.pi/4, np.pi/2, 3*np.pi/4, np.pi, np.pi+np.pi/4, np.pi+np.pi/2, 2*np.pi-np.pi/4]
        fractionplot = 20
        lignetheta0 = np.zeros((size, len(thetaefflist), 3))
        for i in range(0,size):
            for j in range(len(thetaefflist)):
                thetaefflist[j] -= xidev[i]*ds/2/np.pi

            P = generate_circle_by_vectors(t, X[i,:]-X[int(np.size((X[:,0]))/2),:],.2, np.transpose(T[i,:]),  np.transpose(B[i,:])   )
            for j in range(len(thetaefflist)):
                indtheta = np.argmin((t-thetaefflist[j])**2)
                lignetheta0[i,j,0] = P[indtheta, 0]
                lignetheta0[i,j,1] = P[indtheta, 1]
                lignetheta0[i,j,2] = P[indtheta, 2]
                if i > 0 and j > 0 and i % fractionplot == 0:
                    x = np.array([[lignetheta0[i-fractionplot,j-1,0], lignetheta0[i-fractionplot,j,0]], [lignetheta0[i,j-1,0], lignetheta0[i,j,0]]])
                    y = np.array([[lignetheta0[i-fractionplot,j-1,1], lignetheta0[i-fractionplot,j,1]], [lignetheta0[i,j-1,1], lignetheta0[i,j,1]]])
                    z = np.array([[lignetheta0[i-fractionplot,j-1,2], lignetheta0[i-fractionplot,j,2]], [lignetheta0[i,j-1,2], lignetheta0[i,j,2]]])
                    xp = np.array([lignetheta0[i-fractionplot,j-1,0], lignetheta0[i,j-1,0]])
                    yp = np.array([lignetheta0[i-fractionplot,j-1,1], lignetheta0[i,j-1,1]])
                    zp = np.array([lignetheta0[i-fractionplot,j-1,2], lignetheta0[i,j-1,2]])
                    ax3d.plot_surface(x, z, y, color = 'gray', shade = True, edgecolor='k', linewidth = 0.1)
                    #ax3d.plot(xp, yp,zp,  'g-', linewidth = 10)

                    #light = LightSource(90, 45)
                    #illuminated_surface = light.shade(z, cmap=cm.coolwarm)
            if i > 0  and i % fractionplot == 0:
                x = np.array([[lignetheta0[i-fractionplot,-1,0], lignetheta0[i-fractionplot,0,0]], [lignetheta0[i,-1,0], lignetheta0[i,0,0]]])
                y = np.array([[lignetheta0[i-fractionplot,-1,1], lignetheta0[i-fractionplot,0,1]], [lignetheta0[i,-1,1], lignetheta0[i,0,1]]])
                z = np.array([[lignetheta0[i-fractionplot,-1,2], lignetheta0[i-fractionplot,0,2]], [lignetheta0[i,-1,2], lignetheta0[i,0,2]]])
                #xp = np.array([lignetheta0[i-20,-1,0], lignetheta0[i,-1,0]])
                #yp = np.array([lignetheta0[i-20,-1,1], lignetheta0[i,-1,1]])
                ax3d.plot_surface(x, z, y, color = 'gray', shade = True, edgecolor='k' , linewidth = 0.1)
                #ax3d.plot(xp, yp, 'g-')
         #   if i  % 20 == 0: 
                #ax3d.plot(P[:,0], P[:,1], P[:,2], 'k-')

        #lignetheta0 = np.array(lignetheta0)
        #for j in range(len(thetaefflist)):
            #if j == 3:
                #ax3d.plot(lignetheta0[:, j,0], lignetheta0[:,j,1], lignetheta0[:,j,2], 'r', linewidth = 1)
            #else:
                #ax3d.plot(lignetheta0[:, j,0], lignetheta0[:,j,1], lignetheta0[:,j,2], 'k', linewidth = 1)
        ax3d.axis('off')
        ax3d.set_box_aspect((1,1,1), zoom = 4)

        ax3d.view_init(elev=0, azim=66)
        ax3d.set_zlim3d(np.min(X[:,2]-X[int(np.size((X[:,0]))/2),2]), np.max(X[:,2]-X[int(np.size((X[:,0]))/2),2]))
        ax3d.set_ylim3d(np.min(X[:,2]-X[int(np.size((X[:,0]))/2),2]), np.max(X[:,2]-X[int(np.size((X[:,0]))/2),2]))
        ax3d.set_xlim3d(np.min(X[:,2]-X[int(np.size((X[:,0]))/2),2]), np.max(X[:,2]-X[int(np.size((X[:,0]))/2),2]))
        #plt.savefig('PERVERSION_GAMMA_'+str(Gamma)+'_Lambda_'+str(Lambda)+'_N0_'+str(numberofcoils)+'_T_'+str(axialload)+'.pdf' , transparent=True)
        BIGtable = np.zeros((np.size(manifold[:,0]), 11))
        #BIGtable[0,:] = np.array(['Xx', 'Xy', 'Xz', 'kappa1', 'kappa2', 'kappa3', 'Gamma', 'Lambda', 'N_0', 'Tz', 's_res'])
        BIGtable[:,9] = axialload
        BIGtable[:,8] = numberofcoils

        BIGtable[:,10] = tres
        BIGtable[:,7] = Lambda
        BIGtable[:,6] = Gamma
        BIGtable[:,5] = manifold[:,2]
        BIGtable[:,4] = manifold[:,1]
        BIGtable[:,3] = manifold[:,0]
        BIGtable[:,2] = X[:,2]
        BIGtable[:,1] = X[:,1]
        BIGtable[:,0] = X[:,0]


        #np.savetxt('PERVERSION_GAMMA_'+str(Gamma)+'_Lambda_'+str(Lambda)+'_N0_'+str(numberofcoils)+'_T_'+str(axialload)+'.txt', BIGtable )
        #plt.show()
        #plt.savefig('PERVERSION_GAMMA_'+str(Gamma)+'_Lambda_'+str(Lambda)+'_N0_'+str(numberofcoils)+'_T_'+str(axialload)+'.obj' , transparent=True)
        #plt.pause(0.1)
        #plt.close()
    plt.show()

translation_perversion()


