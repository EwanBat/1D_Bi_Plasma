Système à résoudre:
- Equation 1: dn_s1/dt + (n_s0 + n_s1) dx/du_s1 + u_s1 dn_si/dx = 0
- Définition: (1/(n_s*m_s)) * dP_s/dx = P_s0 / n_s0 [gamma_s / n_s0 dn_s1/dx (1 + gamma_s n_s1/n_s0) - 2 gamma_s / n_s0^2 n_s1 dn_s1/dx ] 
- Equation 2: du_s1/dt + u_s1 du_s1/dx + (1/(n_s*m_s)) * dP_s/dx = q_s E / m_s
- Equation 3: dE/dx = 1/epsilon_0 (q_i n_i + q_e n_e)

avec s = i (ions) ou e (électrons)
Constantes connues:
- m_s: masse de l'espèce s
- q_s: charge de l'espèce s
- P_s0: pression de fond de l'espèce s
- n_s0: densité de fond de l'espèce s
- gamma_s: coefficient adiabatique de l'espèce s
- epsilon_0: permittivité du vide

Variables à résoudre:
- n_s1(x,t): perturbation de la densité de l'espèce s
- u_s1(x,t): perturbation de la vitesse de l'espèce s
- E(x,t): champ électrique

Système à 5 équations couplées pour n_i1, u_i1, n_e1, u_e1 et E. Pour chaque espèce s, les équations 1 et 2 décrivent la dynamique de la densité et de la vitesse, tandis que l'équation 3 relie le champ électrique aux densités des espèces chargées.

Pour le moment résoudre avec des méthodes numériques simples (ex: différences finies explicites) en 1D spatiale et temporelle.
But: Simuler l'évolution temporelle des perturbations de densité et de vitesse des ions et des électrons ainsi que du champ électrique dans un plasma.