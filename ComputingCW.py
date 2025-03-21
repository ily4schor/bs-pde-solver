# Dear GTA, welcome to our great code, let us guide you through it.

# First you will find the Control Box, where you can specify all the parameters. 
# You can also specify method specific parameters h,k,M or N for both methods. 
# Then you can choose the method(s) you want to run, and the plots you want to generate.
# Do not miss the special animation, as it is not shown on the word document
# Finally, you can also choose if you want to plot the results at one specific node vs number of spatial nodes, and pick the parameters for this plot.

# After the Control Box, there are the Functions Definition Section (Line 57) and the Functions Calling Section (Line 339) which have their own explanations headers.

# We know the structure is not exactly like in the template, sorry about that, but we tried to make it more concise and easy to use
# Have a good time marking our code!

##CONTROL BOX##

#Market parameters:
vol = 0.5 #Volatility
r = 0.05 #Risk free return rate

#Option specific parameters:
K = 100 #Strike price
T = 1 #Time to expiration

#Method specific parameters for the explicit method (if needed):
h_exp = None #Space step
M_exp = None #Spatial nodes
k_exp = None #Time step
N_exp = None #Time nodes
#DO NOT specify more than one, the function will automatically calculate the other parameters for maximum resolution or minimum computational cost while ensuring convergence and stability.

#Method specific parameters for the implicit method (if needed):
h_imp = None #Space step
M_imp = None #Spatial nodes
k_imp = None #Time step
N_imp = None #Time nodes
#DO NOT specify k and N or h and M

#What method do you want to use?
Explicit = True
Implicit = True

#What plots do you want to show?
Surface_Plot = True
Heat_Map = True
Contour_Plot = True
Animation = True

#Do you want to plot the results for a specific point against the number of spatial nodes?
Results_vs_M = True

#Parameters of the plot
SCheck = 95 #Which point to choose? Only the S can be picked, the tau is automatically at the maximum value.
N = 400 #Do not go too high, this is computationally expensive
Mmax = 137 #Be careful, the Courant condition can be violated for the explicit method


##FUNCTIONS DEFINITION SECTION##
#In this section are defined all the necessary functions
#3 general functions are defined first, because they are used by both methods
#Then the two main functions are defined, one for each method. This is where the methods are implemented.
#The explicit function automatically calculates the other parameters for maximum resolution or minimum computational cost while ensuring convergence and stability, but if you want to see what happens when the Courant condition is violated, you can go in the function and change the 0.99 into 1.1 or the 1.01 into 0.9.
#After that, the functions for the 4 differents visual representations of the solution are defined.
#Finally, the 2 functions to plot the results for a specific node vs number of spatial nodes are defined. The explicit one is a little longer as we had to rewrite the main function without automatically calculating the parameters.

# In this section I am importing all the libraries I will need
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import matplotlib.animation as anim

#The steps in common for both methods are written as separate functions
def yLimits(K): #Function that defines the price limits for which we are checking
    Smin = K/10 #Minimum stock price we are checking for (10 times less than strike price)
    Smax = 3*K #Maximum stock price we are checking for (3 times the strike price)

    #The substitution y = ln(S) is discussed in the word document
    ymin = np.log(Smin)
    ymax = np.log(Smax)
    return ymin, ymax

def setUpGrids(ymin,ymax,T,k,h): #Function that sets up all the necessary grids to compute and store the solution
    #Time domain
    tau = np.arange(0, T+k, k) #The tau = T-t substitution is also discussed in the word document
    N = len(tau) #Creating a variable the length of the time array, useful for matrix size later
    #Price domain - spatial variable 
    y = np.arange(ymin, ymax+h, h)
    S = np.exp(y) #Setting the actual price array, with the substitution reversed
    M = len(y) #Creating a variable the length of the space array, useful for matrix size later
    # Setting up the 2D grid
    C = np.zeros((M, N)) #Empty matrix that will get the price of the Call option
    tauG, yG = np.meshgrid(tau, y) #Grids of tau and y
    SG = np.exp(yG) #Grid of asset price, with the reversed substitution
    return tauG, yG, SG, C, M, N

def setUpBC_IC(C,SG,tauG,K,r): #Function that sets up the boundary conditions and initial values (more like terminal values but we are going back in time)
    #BC
    C[0,:] = 0 #If the asset price falls near zero (in this case Smin), the price of a call option is 0
    C[-1,:] = SG[-1,-1]- K*np.exp(-r*tauG[-1,:]) #if the asset price tends to infinity (in this case 3*K), the BC chosen is that the call option's price tends to the Smax-K*e**(-r*tau). There are other options but we chose this one.

    #IV or more like TV
    C[:,0] = np.maximum(SG[:,0]-K,0) #At expiration time, the call option is worth S-K if the strike price is lower than the asset price, or 0 if it is higher.

#Two different functions for two different methods
def explicitBSCall(K,T,vol,r,k=None,h=None,M=None,N=None):
    #This section exists because the caller can specify one of the k,h,M or N parameter, while the others are specifically calculated to ensure convergence and maximum resolution
    if sum(x is not None for x in [k,h,M,N]) >=2: #If the caller specifies more than one parameter, an error message appears.
        print('Please only specify one of k,h,M,N, the others will be specifically calculated to ensure convergence and maximum resolution')
    else:
        #In this section we are setting the domain of solution and the discretised grid
        ymin, ymax = yLimits(K)

        #Setting up price step h and time step k
        if h is None and M is None:
            if k is None and N is None:
                k = T/1000 #if no variable is specified we are using 1000 time nodes
            if N is not None:
                k = T/N
            hmin = vol*k**0.5
            h = 1.01*hmin
        elif k is None and N is None:
            if M is not None:
                h = (ymax-ymin)/M
            kmax = h**2/vol**2
            k = 0.99*kmax

        #Finally creating the arrays 
        tauG, yG, SG, C, M, N = setUpGrids(ymin,ymax,T,k,h)

        #In this section we are setting up the boundary conditions and initial values
        setUpBC_IC(C,SG,tauG,K,r)

        #In this section we are defining the matrix constants. The derivation is in the word document.
        a = (0.5*k/h)*(vol**2/h-r+vol**2/2)
        b = 1-vol**2*k/h**2-k*r
        c = (0.5*k/h)*(vol**2/h+r-vol**2/2)

        #In this section we are creating the explicit matrix
        M1 = np.zeros((M-2, M)) #Creating the empty matrix with the correct dimensions.
        for i in range(0,M-2): #Setting up the numbers in the diagonal matrix
            M1[i,i] = a
            M1[i,i+1] = b
            M1[i,i+2] = c

        #In this section we are implementing the explicit method 
        for n in range(0,N-1): #Solving the system for every time step
            C[1:-1,n+1] = np.dot(M1,C[:,n]) #Solving the matrix multiplication

        return tauG, SG, C,M, N #returning the price and time to expiration grids with the price matrix, along with the number of nodes in space and time.
    
def implicitBSCall(K,T,vol,r,k=None,h=None,M=None,N=None):
    if (h is not None and M is not None) or (k is not None and N is not None): #If the caller specifies more than one parameter, an error message appears.
        print('You cannot specify both steps and number of nodes in the same dimension')
    else:
        #In this section we are setting the domain of solution and the discretised grid
        ymin, ymax = yLimits(K)

        #Setting up price step h
        if h is None and M is None:
            h = (ymax-ymin)/1000 #if neither h nor M are specified, we are using 1000 nodes
        elif M is not None:
            h = (ymax-ymin)/M
        
        #Setting up time step k
        if k is None and N is None:
            k = T/1000 #is neither k nor N are speciified, we are using 1000 nodes
        elif N is not None:
            k = T/N

        #Finally creating the arrays 
        tauG, yG, SG, C, M, N = setUpGrids(ymin,ymax,T,k,h)

        #In this section we are setting up the boundary conditions and initial values
        setUpBC_IC(C,SG,tauG,K,r)

        #In this section we are defining the matrix constants. The derivation is in the word document.
        a = 0.25*k*((r-vol**2/2)/h - (vol**2)/h**2)
        b = 1+0.5*k*((vol**2)/h**2 + r)
        c = -0.25*k*((r-vol**2/2)/h + (vol**2)/h**2)
        d = 1-0.5*k*(vol**2)/h**2 - 0.5*k*r

        #In this section we are creating the implicit matrix
        M1 = np.zeros((M-2, M-2)) #Creating the empty matrix with the dimensions of the inside points i.e. those not on boundaries

        for i in range(1,M-3): #Setting the numbers in the diagonal matrix
            M1[i,i-1] = a
            M1[i,i] = b
            M1[i,i+1] = c
        M1[0,0] = b #Setting up the first and last row
        M1[0,1] = c
        M1[-1,-1] = b
        M1[-1,-2] = a

        #In this section we are creating the explicit matrix
        M2 = np.zeros((M-2, M-2)) #Creating the empty matrix with the dimensions of the inside points i.e. those not on boundaries

        for i in range(1,M-3): #Setting the numbers in the diagonal matrix
            M2[i,i-1] = -a
            M2[i,i] = d
            M2[i,i+1] = -c
        M2[0,0] = d #Setting up the first and last row
        M2[0,1] = -c
        M2[-1,-1] = d
        M2[-1,-2] = -a

        #In this section we are implementing the Crank Nicolson method, i.e. solving the system for all points
        M1inv = la.inv(M1) #Because of the y substitution, the implicit matrix has constant coefficients and can be inverted only once, outside the loop
        for n in range(0,N-1): #Solving the system for every time step
            RHS = np.dot(M2,C[1:-1,n]) #Defining the right hand side and solving the first matrix multiplication
            RHS[-1] += -c*(C[-1,n]+C[-1,n+1]) #Applying the boundary conditions
            C[1:-1,n+1] = M1inv.dot(RHS) #Solving the last matrix multiplication

        return tauG, SG, C, M, N #returning the price and time to expiration grids with the price matrix, along with the number of nodes in space and time.
    
def surfacePlot(arrays, title='Surface Plot'): #Plots the results using a surface plot
    x,y,z = arrays
    plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(x,y,z,cmap='plasma')
    ax.set_xlabel("Time to expiration (years)")
    ax.set_ylabel("Stock Price")
    ax.set_zlabel("Call Option Price")
    plt.title(title)
    plt.tight_layout()

def heatMap(arrays,title='Heat Map'): #Plots the results using a heat map
    x,y,z = arrays
    plt.figure()
    plt.pcolormesh(x,y,z, shading='auto', cmap='plasma')
    plt.xlabel("Time to expiration (years)")
    plt.ylabel("Asset Price")
    plt.colorbar(label="Call Option Price")
    plt.title(title)
    plt.tight_layout()

def contourPlot(arrays,title='Contour plot',manual=False): #Plots the results using a Contour Plot. The user can choose to click on the lines to reveal the labels by setting manual=True
    x,y,z = arrays   
    plt.figure()
    CP = plt.contour(x,y,z,levels=20,cmap='plasma')
    plt.xlabel("Time to expiration (years)")
    plt.ylabel("Asset Price")
    plt.title(title)
    plt.clabel(CP, inline=True, manual=manual)
    plt.tight_layout()

def animation(arrays,title='Animation'): #Plots the results by creating an animation of the option price against asset price over time, all the way up until the expiration time
    t,y,z = arrays
    fig, ax = plt.subplots()
    line, = ax.plot(y[:,0], z[:, 0], color='purple')
    ax.set_xlabel("Asset Price")
    ax.set_ylabel("Call Option Price")
    ax.set_title(title+f"{title} Time to expiration = {t[0,-1]:.2f} years")
    ax.set_xlim(y.min(), y.max())
    ax.set_ylim(z.min(), z.max())
    def update(frame):
        line.set_ydata(z[:, -frame])
        ax.set_title(f"{title} Time to expiration = {t[0,-frame]:.2f} year(s)")
        return (line,)
    ani = anim.FuncAnimation(
        fig,           
        update,        
        frames=range(len(t[0,:])),  
        interval=50,  #The delay between frames in milliseconds can be controlled here
    )
    return ani

def expCheckResultsVsM(K,T,vol,r,N,Mmax,SCheck):  #Function that plots the results for a specific node vs number of spatial nodes for the explicit method
    Ms = range(3,Mmax,1) #Defining the range of number of nodes we are testing for

    def badExplicitBSCall(K,T,vol,r,M,N): #New function with all the calculations at the beginning removed.
        #In this section we are setting the domain of solution and the discretised grid
        ymin, ymax = yLimits(K)
        k = T/N
        h = (ymax-ymin)/M

        #Creating the arrays 
        tauG, yG, SG, C, M, N = setUpGrids(ymin,ymax,T,k,h)

        #In this section we are setting up the boundary conditions and initial values
        setUpBC_IC(C,SG,tauG,K,r)

        #In this section we are defining the matrix constants. The derivation is in the word document.
        a = (0.5*k/h)*(vol**2/h-r+vol**2/2)
        b = 1-vol**2*k/h**2-k*r
        c = (0.5*k/h)*(vol**2/h+r-vol**2/2)

        #In this section we are creating the explicit matrix
        M1 = np.zeros((M-2, M)) #Creating the empty matrix with the correct dimensions.
        for i in range(0,M-2): #Setting up the numbers in the diagonal matrix
            M1[i,i] = a
            M1[i,i+1] = b
            M1[i,i+2] = c

        #In this section we are implementing the explicit method 
        for n in range(0,N-1): #Solving the system for every time step
            C[1:-1,n+1] = np.dot(M1,C[:,n]) #Solving the matrix multiplication

        return tauG, SG, C,M, N #returning the price and time to expiration grids with the price matrix, along with the number of nodes in space and time.

    oPricesExp = np.array([]) #Initialising the y array
    nodesNumberExp = np.array([]) #Initialising the x array
    for m in Ms: 
        ex = badExplicitBSCall(K,T,vol,r,m,N) #Running the method for each different M
        slice = ex[2][:,-1] #Taking the last time slice for the option prices
        aPrice = ex[1][:,3] #Taking any slice of asset price (does not vary)
        target = SCheck 
        oPriceTarget = np.interp(target, aPrice, slice) #Find the value for the specific asset price we are looking for, by interpolation if it is not precisely on a node.
        nodesNumberExp = np.append(nodesNumberExp, ex[3]) #Adding the real M value to the x array
        oPricesExp = np.append(oPricesExp,oPriceTarget) #Adding the option price value to the y array
    plt.figure()
    plt.plot(nodesNumberExp,oPricesExp) #Plotting them against each other
    plt.xlabel("Number of spatial nodes M")
    plt.ylabel("Call Option Price")
    plt.title(f"Explicit method, for node S={SCheck}, tau={T}")
    plt.tight_layout()

def impCheckResultsVsM(K,T,vol,r,N,Mmax,SCheck): #Function that plots the results for a specific node vs number of spatial nodes for the explicit method
    Ms = range(3,Mmax,1) #defining the range of number of nodes we are testing for

    oPricesImp = np.array([]) #Initialising the y array
    nodesNumberImp = np.array([]) #Initialising the x array
    for m in Ms:
        im = implicitBSCall(K,T,vol,r,M=m,N=N) #Running the method for each different M
        slice = im[2][:,-1] #Taking the last time slice for the option prices
        aPrice = im[1][:,3] #Taking any slice of asset price (does not vary)
        target = SCheck
        oPriceTarget = np.interp(target, aPrice, slice) #Find the value for the specific asset price we are looking for, by interpolation if it is not precisely on a node.
        nodesNumberImp = np.append(nodesNumberImp, im[3]) #Adding the real M value to the x array
        oPricesImp = np.append(oPricesImp,oPriceTarget) #Adding the option price value to the y array
    plt.figure()
    plt.figure()
    plt.plot(nodesNumberImp,oPricesImp) #Plotting them against each other
    plt.xlabel("Number of spatial nodes M")
    plt.ylabel("Call Option Price")
    plt.title(f"Implicit method, for node S={SCheck}, tau={T}")
    plt.tight_layout()
    plt.show()


##FUNCTIONS CALLING SECTION##
#This section is pretty straightforward, the logic comes from what was defined in the control box.
#The actual number of nodes for both method is printed in the terminal

if Explicit == True:
    exp = explicitBSCall(K,T,vol,r,k=k_exp,h=h_exp,M=M_exp,N=N_exp)
    print('Explicit Method nodes: (space x time)', exp[3], 'x', exp[4])
    if Surface_Plot == True:
        surfacePlot(exp[0:3], title='Surface Plot for the Explicit Method')
    if Heat_Map == True:
        heatMap(exp[0:3], title='Heat Map for the Explicit Method')
    if Contour_Plot == True:
        contourPlot(exp[0:3],title='Contour Plot for the Explicit Method')
    if Animation == True:
        ani = animation(exp[0:3],title='Explicit method,')
    if Results_vs_M == True:
        expCheckResultsVsM(K,T,vol,r,N,Mmax,SCheck)


if Implicit == True:
    imp = implicitBSCall(K,T,vol,r,k=k_imp,h=h_imp,M=M_imp,N=N_imp)
    print('Implicit Method nodes: (space x time)', imp[3], 'x', imp[4])
    if Surface_Plot == True:
        surfacePlot(imp[0:3], title='Surface Plot for the Implicit Method')
    if Heat_Map == True:
        heatMap(imp[0:3], title='Heat Map for the Implicit Method')
    if Contour_Plot == True:
        contourPlot(imp[0:3],title='Contour Plot for the Implicit Method')
    if Animation == True:
        ani = animation(imp[0:3],title='Implicit method,')
    if Results_vs_M == True:
        impCheckResultsVsM(K,T,vol,r,N,Mmax,SCheck)

plt.show()