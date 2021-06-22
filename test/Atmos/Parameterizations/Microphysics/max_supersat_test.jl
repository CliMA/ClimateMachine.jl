@testset "max_supersat_test" begin
    # parameters inputted into function:
    a_mi = 5*(10^(-8)) # particle mode radius (m)
    sigma_i = 2 # standard deviation of mode radius (m)
    tau = 1 # time of activation (s)                                                  
    M_w = 0.01801528 # Molecular weight of water (kg/mol)
    rho_w = 1000 # Density of water (kg/m^3)
    R = 8.31446261815324 # Gas constant (kg*m^2/s^2*K*mol)
    T = 273.15 # Temperature (K)
    B_i_bar = 1 # calculated in earlier function 
    alpha = 1 # Coefficient in superaturation balance equation       
    V = 1 # Updraft velocity (m/s)
    G = 1 # Diffusion of heat and moisture for particles 
    N_i = 100000000 # Initial particle concentration (1/m^3)
    gamma = 1 # coefficient 

    # Internal calculations:
    f_i = 0.5 * exp(2.5*(log(sigma_i))^2) # function of sigma_i (check units)
    g_i = 1 + 0.25 * log(sigma_i) # function of sigma_i (log(m))
    A = (2*tau*M_w)/(rho_w*R*T) # Surface tension effects on Kohler equilibrium equation (s/(kg*m))
    S_min = ((2)/(B_i_bar)^(.5))((A)/(3*a_mi))^(3/2) # Minimum supersaturation
    zeta = ((2*A)/(3))*((alpha*V)/(G))^(.5) # dependent parameter 
    eta = (  ((alpha*V)/(G))^(3/2)  /    (2*pi*rho_w*gamma*N_i)   )    # dependent parameter

    
    # Final value for maximum supersaturation:
    MS = ((1)/(S_min)^2) * (    f_i*((zeta/eta)^(3/2))     +    g_i*(((S_min^2)/(eta+3*zeta))^(3/2))     )

    # Comaparing calculated MS value to function output: 
    @test maxsupersat(a_mi, sigma_i tau, M_w, rho_w, R, T, B_i_bar, alpha, V, G, N_i, gamma) = MS
end