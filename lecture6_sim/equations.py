"""Display equations for every lecture experiment, using standard LaTeX.

Entries are short display-math lines without dollar delimiters.  They are kept
separate from the searchable plain-text topic descriptions and require an
amsmath-capable renderer (not plain-text drawing or a restricted math parser).
"""


LATEX_EQUATIONS: dict[int, tuple[str, ...]] = {
    26: (
        r"R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\"
        r"\sin\theta&\cos\theta\end{pmatrix}",
        r"\theta\in\mathbb{S}^{1},\qquad \theta\equiv\theta+2\pi",
    ),
    27: (
        r"\bar\theta=\operatorname{atan2}\!\left("
        r"\mathbb{E}[\sin\theta],\mathbb{E}[\cos\theta]\right)",
        r"R=\left|\mathbb{E}\!\left[e^{i\theta}\right]\right|",
    ),
    28: (
        r"\rho_{\mathrm W}(\theta)=\sum_{k\in\mathbb Z}"
        r"\mathcal N\!\left(\theta-2\pi k;\mu,\sigma^2\right)",
        r"\rho_{\mathrm W}(\theta+2\pi)=\rho_{\mathrm W}(\theta)",
    ),
    29: (
        r"\frac{\partial f}{\partial t}="
        r"\frac{\kappa}{2}\frac{\partial^2 f}{\partial\theta^2},"
        r"\qquad f(\theta+2\pi,t)=f(\theta,t)",
    ),
    30: (
        r"X=F^{-1}(U),\qquad \mathbf q=\frac{\mathbf Z}{\|\mathbf Z\|}"
        r",\quad \mathbf Z\sim\mathcal N(\mathbf0,I_3)",
        r"\mathbf q=\frac{\mathbf v}{\|\mathbf v\|},\qquad"
        r"\text{accept }\mathbf v\ \Longleftrightarrow\ 0<\|\mathbf v\|\le1",
    ),
    31: (
        r"U\sim\mathcal U(0,1),\qquad X=F^{-1}(U)",
        r"X=-\frac{\ln(1-U)}{\lambda}\sim\operatorname{Exp}(\lambda)",
    ),
    32: (
        r"\theta=\arccos(1-2U),\qquad\phi=2\pi V",
        r"U,V\overset{\mathrm{iid}}{\sim}\mathcal U(0,1),"
        r"\qquad\mathbb E[\mathbf q\mathbf q^{\mathsf T}]=\frac{I_3}{3}",
    ),
    33: (
        r"p_Y(y)=p_X\!\left(\psi^{-1}(y)\right)"
        r"\left|\det D\psi^{-1}(y)\right|",
        r"Y=\psi(X),\qquad\psi\text{ invertible}",
    ),
    34: (
        r"\begin{pmatrix}Z_1\\Z_2\end{pmatrix}="
        r"\sqrt{-2\ln U}\begin{pmatrix}\cos(2\pi V)\\\sin(2\pi V)\end{pmatrix}",
        r"U,V\overset{\mathrm{iid}}{\sim}\mathcal U(0,1),"
        r"\qquad Z_1,Z_2\overset{\mathrm{iid}}{\sim}\mathcal N(0,1)",
    ),
    35: (
        r"\mathbf X=\boldsymbol\mu+L\mathbf Z,\qquad LL^{\mathsf T}=\Sigma",
        r"\mathbf Z\sim\mathcal N(\mathbf0,I),"
        r"\qquad\mathbf X\sim\mathcal N(\boldsymbol\mu,\Sigma)",
    ),
    36: (
        r"\mathbf v\sim\mathcal U([-1,1]^3),\qquad"
        r"\text{accept}\ \Longleftrightarrow\ 0<\|\mathbf v\|\le1",
        r"\mathbf q=\frac{\mathbf v}{\|\mathbf v\|}\in\mathbb S^2",
    ),
    37: (
        r"\mathbb P(\mathrm{accept})="
        r"\frac{\pi^{d/2}}{2^d\Gamma\!\left(\frac d2+1\right)}",
        r"d=3:\qquad\mathbb P(\mathrm{accept})=\frac{\pi}{6}",
    ),
    38: (
        r"K_{n+1}=K_n+\xi_n,\qquad K_0=0",
        r"\mathbb P(\xi_n=1)=\mathbb P(\xi_n=-1)=\frac12"
        r",\qquad\xi_n\text{ independent}",
    ),
    39: (
        r"p_{n+1}(k)=\frac{p_n(k-1)+p_n(k+1)}{2}",
        r"p_0(k)=\delta_{k0}",
    ),
    40: (
        r"p_n(k)=\begin{cases}"
        r"2^{-n}\binom{n}{(n+k)/2},&|k|\le n,\ k\equiv n\pmod2,\\"
        r"0,&\text{otherwise}.\end{cases}",
    ),
    41: (
        r"\frac{\sum_{i=1}^{n}(X_i-\mu)}{\sigma\sqrt n}"
        r"\xrightarrow{\ d\ }\mathcal N(0,1)",
        r"X_i\text{ iid},\qquad\mathbb E[X_i]=\mu,"
        r"\quad0<\operatorname{Var}(X_i)=\sigma^2<\infty",
    ),
    42: (
        r"\frac{h^2}{\Delta t}=D,\qquad"
        r"\frac{\partial f}{\partial t}=\frac D2\frac{\partial^2 f}{\partial x^2}",
        r"\operatorname{Var}(X_t)=Dt\qquad(X_0=0)",
    ),
    43: (
        r"\Delta W_k=\sqrt{\Delta t}\,Z_k,\qquad "
        r"Z_k\overset{\mathrm{iid}}{\sim}\mathcal N(0,1)",
        r"\operatorname{Var}(\Delta W_k)=\Delta t,\qquad"
        r"\frac{\Delta W_k}{\Delta t}=\frac{Z_k}{\sqrt{\Delta t}}",
    ),
    44: (
        r"W_{k+1}=W_k+\sqrt{\Delta t}\,Z_k,\qquad W_0=0",
        r"Z_k\overset{\mathrm{iid}}{\sim}\mathcal N(0,1),\qquad "
        r"W_t\sim\mathcal N(0,t)",
    ),
    45: (
        r"\mathrm dX_t=h(X_t,t)\,\mathrm dt+H(X_t,t)\,\mathrm dW_t",
        r"\mathrm dX_t=\mu X_t\,\mathrm dt+\sigma X_t\,\mathrm dW_t",
    ),
    46: (
        r"X_{k+1}=X_k+h(X_k,t_k)\Delta t"
        r"+H(X_k,t_k)\sqrt{\Delta t}\,Z_k",
        r"Z_k\overset{\mathrm{iid}}{\sim}\mathcal N(0,I)",
    ),
    47: (
        r"\mathrm dX_t=-\Gamma X_t\,\mathrm dt+C\,\mathrm dW_t",
        r"\mathbb P(X_t\in A)=\int_A f(x,t)\,\mathrm dx",
    ),
    48: (
        r"\frac{\partial f}{\partial t}="
        r"-\nabla\!\cdot(hf)+\frac12\sum_{i,j}"
        r"\frac{\partial^2(B_{ij}f)}{\partial x_i\,\partial x_j}",
        r"B=HH^{\mathsf T}",
    ),
    49: (
        r"\frac{\partial f}{\partial t}+\nabla\!\cdot\mathbf J=0,"
        r"\qquad\mathbf J=hf-\frac12B\nabla f\quad(B\text{ constant})",
        r"\mathbf J\!\cdot\mathbf n=0\ \text{on }\partial\Omega"
        r"\quad\Longrightarrow\quad\frac{\mathrm d}{\mathrm dt}"
        r"\int_\Omega f\,\mathrm dx=0",
    ),
    50: (
        r"m\ddot x+c\dot x+kx=F(t)",
        r"F(t)\,\mathrm dt=q\,\mathrm dW_t",
    ),
    51: (
        r"\mathrm d\begin{pmatrix}x\\v\end{pmatrix}="
        r"\begin{pmatrix}0&1\\-\frac{k}{m}&-\frac{c}{m}\end{pmatrix}"
        r"\begin{pmatrix}x\\v\end{pmatrix}\mathrm dt+"
        r"\begin{pmatrix}0\\\frac{q}{m}\end{pmatrix}\mathrm dW_t",
    ),
    52: (
        r"\mathrm d\mathbf X_t=-\Gamma\mathbf X_t\,\mathrm dt+C\,\mathrm d\mathbf W_t",
        r"B=CC^{\mathsf T}",
    ),
    53: (
        r"\dot{\boldsymbol\mu}=-\Gamma\boldsymbol\mu",
        r"\dot\Sigma=-\Gamma\Sigma-\Sigma\Gamma^{\mathsf T}+B",
    ),
    54: (
        r"\Gamma\Sigma_\infty+\Sigma_\infty\Gamma^{\mathsf T}=B",
        r"\mathbf X_\infty\sim\mathcal N(\mathbf0,\Sigma_\infty),"
        r"\qquad\operatorname{Re}\lambda_i(\Gamma)>0",
    ),
    55: (
        r"\Sigma_\infty=\int_0^{\infty}"
        r"e^{-\Gamma t}B\,e^{-\Gamma^{\mathsf T}t}\,\mathrm dt",
        r"\operatorname{Re}\lambda_i(\Gamma)>0",
    ),
    56: (
        r"\Gamma=U\Lambda V^{\mathsf T},\qquad V^{\mathsf T}U=I",
        r"\Gamma u_i=\lambda_i u_i,\qquad "
        r"v_i^{\mathsf T}\Gamma=\lambda_i v_i^{\mathsf T}",
    ),
    57: (
        r"\Sigma_\infty=\sum_{i,j}"
        r"\frac{v_i^{\mathsf T}Bv_j}{\lambda_i+\lambda_j}"
        r"u_i u_j^{\mathsf T}",
        r"V^{\mathsf T}U=I,\qquad\operatorname{Re}\lambda_i>0",
    ),
    58: (
        r"\sigma^2(t)=e^{-2\gamma t}\sigma_0^2+"
        r"\frac{c^2}{2\gamma}\left(1-e^{-2\gamma t}\right)",
        r"\sigma_\infty^2=\frac{c^2}{2\gamma},\qquad\gamma>0",
    ),
    59: (
        r"p\!\left(\mathbf x_t,\mathbf m\mid z_{1:t},u_{1:t}\right)",
        r"\mathbf x_t=(x_t,y_t,\theta_t)^{\mathsf T},\qquad"
        r"\mathbf m=(\ell_1^{\mathsf T},\ldots,\ell_M^{\mathsf T})^{\mathsf T}",
    ),
    60: (
        r"\mathrm d\mathbf X_t=-\Gamma\mathbf X_t\,\mathrm dt+C\,\mathrm d\mathbf W_t"
        r"\quad\Longleftrightarrow\quad\partial_t f=\mathcal L^*f",
        r"\Gamma\Sigma_\infty+\Sigma_\infty\Gamma^{\mathsf T}=CC^{\mathsf T}",
    ),
    61: (
        r"\operatorname{Cov}(Z_i,Z_j)=\delta_{ij},\qquad"
        r"\Delta W=\sqrt{\Delta t}\,Z",
        r"\partial_t f=\mathcal L^*f,\qquad"
        r"\operatorname{Re}\lambda_i(\Gamma)>0",
    ),
}
