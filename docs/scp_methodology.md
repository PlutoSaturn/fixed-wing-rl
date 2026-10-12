Base Rules

Initialization

Tracked Dynamic Constraints: For the duration:

This solver uses a 3-DOF representation of the defined aircraft parameters

Derived Dynamic Constraints:

Modeling of Obstacles

Method of Solution

## Aircraft dynamics

**State and control**

$$\mathbf{x} = [p_x,\ p_y,\ p_z,\ V,\ \gamma,\ \chi]^\top, \qquad \mathbf{u} = [T,\ n_v,\ n_h]^\top$$

**Constants**

$$a_1 = \tfrac{1}{2}\rho S C_{D_0}, \qquad a_2 = \frac{2K(mg)^2}{\rho S}, \qquad c_{\text{stall}} = \frac{\rho S C_{L,\max}}{2mg}$$

**Drag**

$$D = a_1 V^2 + a_2\,\frac{n_v^2 + n_h^2}{V^2}$$

**Equations of motion**

$$\dot p_x = V\cos\gamma\cos\chi$$

$$\dot p_y = V\cos\gamma\sin\chi$$

$$\dot p_z = V\sin\gamma$$

$$\dot V = \frac{T - D}{m} - g\sin\gamma$$

$$\dot\gamma = \frac{g\,(n_v - \cos\gamma)}{V}$$

$$\dot\chi = \frac{g\,n_h}{V\cos\gamma}$$

**Derived quantities**

$$n = \sqrt{n_v^2 + n_h^2}, \qquad \mu = \operatorname{atan2}(n_h, n_v), \qquad R = \frac{V^2}{g\tan\mu}$$

**Free final time**

$$\frac{d\mathbf{x}}{d\tau} = \sigma f(\mathbf{x},\mathbf{u}), \qquad \tau \in [0,1]$$

**Constraints**

$$0 \le T \le f_T\,(T_0 + T_1 V)$$

$$\sqrt{n_v^2 + n_h^2} \le N_{\max}$$

$$|n_h| \le \tan(\mu_{\max})\, n_v$$

$$\sqrt{n_v^2 + n_h^2} \le c_{\text{stall}} V^2 \quad \text{(linearized: } \le c_{\text{stall}}\,(2\bar V V - \bar V^2)\text{)}$$

$$V_{\min} \le V \le V_{\max}, \qquad |\gamma| \le \gamma_{\max}$$

$$|n_{h,k+1} - n_{h,k}| \le \dot n_{h,\max}\,\Delta t_k, \qquad |n_{v,k+1} - n_{v,k}| \le \dot n_{v,\max}\,\Delta t_k$$

## Multiple shooting

$$\mathbf{u}(s) = (1-s)\,\mathbf{u}_k + s\,\mathbf{u}_{k+1}$$

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \int_0^1 \Delta\tau\,\sigma\, f(\mathbf{x}(s),\mathbf{u}(s))\,ds$$

$$\mathbf{x}_{k+1} = \bar F_k + \Phi_k\,\delta\mathbf{x}_k + B^-_k\,\delta\mathbf{u}_k + B^+_k\,\delta\mathbf{u}_{k+1} + S_k\,\delta\sigma + \boldsymbol{\nu}_k$$

$$\frac{d\Phi}{ds} = \Delta\tau\,\sigma A\Phi, \qquad \frac{dB^-}{ds} = \Delta\tau\,\sigma\,(AB^- + (1-s)B)$$

$$\frac{dB^+}{ds} = \Delta\tau\,\sigma\,(AB^+ + sB), \qquad \frac{dS}{ds} = \Delta\tau\,(\sigma AS + f)$$

## Birkhoff

$$\mathbf{x}_i = \mathbf{x}_0 + \sum_j B_{ij}\,\mathbf{V}_j, \qquad B_{ij} = \int_{-1}^{\tau_i} \ell_j(\tau)\,d\tau$$

$$\mathbf{V}_i = \frac{\sigma}{2S}\, f(\mathbf{x}_i,\mathbf{u}_i)$$

## Obstacles

$$\hat{\mathbf{n}}_i^\top(\mathbf{p} - \bar{\mathbf{p}}) + s \ge -\frac{h_i(\bar{\mathbf{p}})}{\lVert\nabla h_i(\bar{\mathbf{p}})\rVert}, \qquad s \ge 0$$
