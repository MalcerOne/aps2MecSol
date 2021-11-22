from funcoesTermosol import *
import numpy as np

def condicoesContorno(matrizGlobal, condicoes):
    if np.shape(matrizGlobal)[1] != 1:
        return np.delete(np.delete(matrizGlobal, list(condicoes[:,0].astype(int)),axis=0), list(condicoes[:,0].astype(int)),axis=1)
    else:
        return np.delete(matrizGlobal, list(condicoes[:,0].astype(int)), axis = 0)

def main():
    nn, N, nm, Inc, nc, F, nr, R = importa('entrada-grupo1.xlsx')
    
    E = Inc[0,2]
    A = Inc[0,3]

    plota(N, Inc)

    # Definir a matriz de conectividade
    matrizConectividade = np.zeros((nm, nn))

    nos_1 = Inc[:,0]
    nos_2 = Inc[:,1]

    for i in range(len(nos_1)):
        no_1 = int(nos_1[i]) - 1
        no_2 = int(nos_2[i]) - 1
        matrizConectividade[i][no_1] = -1
        matrizConectividade[i][no_2] = 1
    
    #print(f"Matriz de conectividade: \n{matrizConectividade}")
    
    # Calcular a matriz de rigidez de cada elemento
    # Montar a matriz de rigidez global [Kg] da trelica
    N_transpose = np.transpose(N)
    matrizMembros = np.matmul(matrizConectividade, N_transpose)
    comprimento = np.zeros((np.shape(matrizMembros)[1], 1))
    numeroElementos = np.shape(matrizMembros)[1]
    matrizConectividadeT = np.transpose(matrizConectividade)
    
    linhasMembros = len(matrizMembros)
    linhasConectividade = len(matrizConectividadeT)
    matrizRigidezGlobal = np.zeros((nn*2, nn*2))
    
    for item in range(0, nm):
        nos1 = Inc[item,0]
        nos2 = Inc[item,1]
        ra = int(Inc[:,0][item])
        x1 = N[0][ra -1]
        x2 = N[0][ra -1]
        y1 = N[1][ra -1]
        y2 = N[1][ra -1]    
        L = ((x1-x2)**2+(y1-y2)**2) ** 1/2
        
        k = E*A/L
        
        membros = matrizMembros[:,item]
        membros.shape = [linhasMembros, 1]
        membros_transpose = np.transpose(membros)

        S = (k * np.matmul(membros, membros_transpose)) / (np.linalg.norm(matrizMembros[:,item])**2)

        conectividade = np.transpose(matrizConectividade)[:,item]
        conectividade.shape = [linhasConectividade, 1]
        conectividade_transpose = np.transpose(conectividade)
        conectividadeT = np.matmul(conectividade, conectividade_transpose)

        Ke = np.kron(conectividadeT, S)
        matrizRigidezGlobal = np.r_[matrizRigidezGlobal, [Ke]]

    for col in range(numeroElementos):
        comprimento[col] = np.linalg.norm(matrizMembros[:,col])

    # Aplicar condicoes de contorno
    mRigidezGlobalCC = condicoesContorno(matrizRigidezGlobal, R)
    vetorGlobalForcasCC = condicoesContorno(F, R)

    # Aplicar um metodo numerico para resolver o sistema de equacoes e obter os deslocamentos nodais
    colunas_rigidez = np.shape(mRigidezGlobalCC)[1]
    linhas_rigidez = np.shape(mRigidezGlobalCC)[0]
    
    x = np.zeros((linhas_rigidez,1)) # chute inicial
    xnew = np.zeros((linhas_rigidez, 1)) 
    tolerancia = 1e-10
    p = 100 # maximo de iteracoes
    
    for i in range(p):
        # Método de Jacobi
        for l in linhas_rigidez:
            xnew[l] = (vetorGlobalForcasCC[l] - xnew[l]) / mRigidezGlobalCC[l][l]
        
        # Erro
        erro = max(abs((xnew-x)/xnew))
        
        # Atualizar
        x = np.copy(xnew)
        
        if erro <= tolerancia:
            break

    # Determinar a deformacao em cada elemento
    # Criar matriz de angulos
    matrizTrigo = np.zeros((numeroElementos, 4))

    for elemento in range(numeroElementos):
        matrizTrigo[elemento, 0] = -matrizConectividade[elemento, 2] # cos
        matrizTrigo[elemento, 1] = -matrizConectividade[elemento, 3] #sen
        matrizTrigo[elemento, 2] = (matrizMembros[0, elemento])/comprimento[elemento] # cos
        matrizTrigo[elemento, 3] = (matrizMembros[1, elemento])/comprimento[elemento] # sen

    # Criar matriz de deslocamentos
    matrizDeslocamentos = np.zeros((len(R) + len(x), 1))
    cond = list(R[:,0].astype(int))
    contador = 0

    for deslocamento in range(len(matrizDeslocamentos)):
        if deslocamento not in cond:
            matrizDeslocamentos[deslocamento] = x[contador]
            contador = contador + 1
    
    deformacoes = np.zeros((len(comprimento), 1))

    for item in range(len(comprimento)):
        deformacoes.append((1/comprimento[item]) * (np.matmul(matrizTrigo[item], matrizDeslocamentos[item])))

    # Determinar a tensao em cada elemento
    tensao = E * deformacoes

    # Determinar as reacoes de apoio
    reacoesDeApoio = np.matmul(matrizRigidezGlobal, matrizDeslocamentos)[cond]

    # Determinar as forcas internas
    forcasInternas = A * tensao

    # Gera um arquivo de output com as reacoes de apoio, matriz de deslocamentos, deformacoes, forcas internas e tensao
    geraSaida("output", reacoesDeApoio, matrizDeslocamentos, deformacoes, forcasInternas, tensao)

if __name__ == '__main__':
    main()