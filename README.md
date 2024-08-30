# PyPerv

PyPerv is a python package allowing to compute for the perversion solutions in a multi-circle rod ( helix with zero pitch ) for different axial loads imposed on the rod, based on a shooting technique of the Kirchhoff rod equilibrium equations. 

Requirements : 


## Codes




## Method 

Defining $\mathbf{X}= (T_1, T_2, T_3, \kappa_1, \kappa_2, \kappa_3)$. At a given helix $\mathbf{X_h}=  (\gamma\tau\kappa, 0,\gamma\tau^2, \kappa,0, \tau)$, the Jacobian of the Kirchhoff equations has eigenvalues  $0,0, \pm\sigma \pm i \omega$. To solve for the orbit linking $\mathbf{X_h}$ to another point, we place one initial point on the unstable manifold of the fixed point $\mathbf{X_h}$, with eigenvector $\mathbf{v_+^{(u)}}$ and we set our intial condition as 
$\mathbf{X}(0)  = \mathbf{X_+} +\epsilon \left(\mathcal{R}e \left(\mathbf{v_+^{(u)}}\right) \cos(\theta) +  \mathcal{I}m \left(\mathbf{v_+^{(u)}}\right) \sin(\theta)\right)$
with $\epsilon$ small($\approx 10^{-4}$) so that the initial condition lies close to the fixed point. And we set the phase $\theta$ so that the trajectory converges towards the desired point. 
