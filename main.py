from funcoesTermosol import *
import numpy as np

def main():
    nn, N, nm, Inc, nc, F, nr, R = importa('entrada.xlsx')
    
    E = Inc[0,2]
    A = Inc[0,3]

    plota(N, Inc)

    # Definir a matriz de conectividade
    matrizConectividade = np.zeros((nm, nn))

    for col in range(len(Inc[:,0])):
        matrizConectividade[int(Inc[:,0] - 1), col] = -1
        matrizConectividade[int(Inc[:,1] - 1), col] = 1
        
    print(matrizConectividade)
    
if __name__ == '__main__':
    main()