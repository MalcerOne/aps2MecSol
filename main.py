import numpy as np
from funcoesTermosol import *

def main():
    [nn, N, nm, Inc, nc, F, nr, R] = importa('entrada-grupo1.xlsx')

    plota(N, Inc)
    
    C = []
    
    for i in range(nm):
        C_i = nn*[0]

        no_1 = int(Inc[i, 0])
        no_2 = int(Inc[i, 1])

        C_i[no_1 - 1] = -1
        C_i[no_2 - 1] = 1
        
        C.append(C_i) 

    C_t = np.transpose(np.array(C))
    M = np.matmul(N, C_t)
    E = Inc[0, 2]
    A = Inc[0, 3]
    lenM = len(M)
    lenC = len(C_t)
    Kg = np.zeros((2*nn, 2*nn))
    
    for i in range(nm):
        x1 = N[0][int(Inc[:,0][i])-1]
        y1 = N[1][int(Inc[:,0][i])-1]
        x2 = N[0][int(Inc[:,1][i])-1]
        y2 = N[1][int(Inc[:,1][i])-1]    
        
        L = (((x1-x2)**2+(y1-y2)**2))**0.5
        
        k = (E*A)/L
        
        m_h = M[:,i]
        m_h.shape = [lenM, 1]
        m_h_t = np.transpose(m_h)
        Se = (k * np.matmul(m_h, m_h_t)) / (np.linalg.norm(M[:,i])**2)

        m2_h = C_t[:,i]
        m2_h.shape = [lenC, 1]
        m2_h_t = np.transpose(m2_h)
        
        m_cXh = np.matmul(m2_h, m2_h_t)
        Ke = np.kron(m_cXh, Se)
        Kg += Ke
    
    Kg_c = np.delete(Kg, R.astype(int),0)
    Kg_c = np.delete(Kg_c, R.astype(int), 1)
    x = np.zeros(Kg_c.shape[0])
    F_c = np.delete(F, R.astype(int))
    U_ar = np.linalg.solve(Kg_c, F_c)

    m_d = np.diag(Kg_c)
    k_d = Kg_c - np.diagflat(m_d)
    
    for i in range(100):
        x2 = (F_c - np.matmul(k_d,x)) / m_d
        error =  max(abs((x2 - x) / x2) )
        if error < (1e-10):
            u_j = x2
            break
        
        u_j = x2
        
    u_j_a = np.zeros((2 * nn, 1))
    i = 0
    
    for c in range(len(u_j_a)):
        if c not in R:
            u_j_a[c] += u_j[i]
            i += 1
            
    u = np.zeros((2 * nn,1))
    i = 0
    
    for c in range(len(u)):
        if c not in R:
            u[c] += U_ar[i]
            i += 1
            
    P = np.matmul(Kg, u)
    P_r = np.zeros((nr, 1))
    
    for i in range(nr):  
        index = int(R[i])
        P_r[i] = P[index]
        
    arr_t, arr_f, arr_d = ([] for i in range(3))
    
    for i in range (nm):    
        m_a = [u[(int(Inc[i, 0]) - 1 ) * 2], u[(int(Inc[i, 0]) - 1 ) * 2 + 1], u[(int(Inc[i, 1]) - 1) * 2], u[int(Inc[i, 1] - 1 ) * 2 + 1]]
        
        x1 = N[0][int(Inc[:,0][i])-1]
        y1 = N[1][int(Inc[:,0][i])-1]
        x2 = N[0][int(Inc[:,1][i])-1]
        y2 = N[1][int(Inc[:,1][i])-1]    
        
        L = (((x1-x2)**2 + (y1-y2)**2))**0.5
        E = Inc[i, 2]
        A = Inc[i,3]
        k = E * A / L
        s = (y2-y1)/ L
        c = (x2-x1)/ L
        C = [-c, -s, c, s]

        dfm = (1/L) * np.matmul(C, m_a)
        st = E * dfm
        frc = A * st
        
        arr_t.append(st)
        arr_f.append(frc)
        arr_d.append(dfm)
        
        
    geraSaida('output-grupo1', P_r, u_j_a, arr_d, arr_f, arr_t)
            
if __name__ == '__main__':
    main()