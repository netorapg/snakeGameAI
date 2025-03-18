import matplotlib.pyplot as plt

plt.ion()  # Ativa modo interativo do Matplotlib

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores, label="Pontuação")
    plt.plot(mean_scores, label="Média")
    
    # Define os limites do eixo Y para não distorcer o gráfico
    plt.ylim(ymin=0)
    
    # Adiciona texto nas pontas das curvas para exibir os valores
    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    # Adiciona legenda e exibe o gráfico
    plt.legend()
    plt.pause(0.1)
