"""
O código desse arquivo realiza o preparo do dataset para obter as informações necessárias.
Primeiramente, foram importadas todas as bibliotecas, classes e funções requeridas.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

"""
O caminho de acesso ao dataset escolhido é armazenado na variável lol e enviado
através da função read_csv para a variável dataset
"""
lol = "..\\datasetLoL\\LeagueofLegends.csv"
dataset = pd.read_csv(lol)

'''
O seaborn é utilizado apenas para tratamentos gráficos
'''
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis') #Verificando a presença de dados nulos

'''
É feito uma cópia do dataset principal para um auxiliar
'''
dsAux = dataset.copy()

'''
O nosso dataframe principal é criado com o nome de 'partidas'
'''
partidas = pd.DataFrame(columns=['tagAzul','ResultadoAzul','OuroAzul','killsAzul','TorresAzul','InibidoresAzul','DragoesAzul','BaroesAzul','ArautosAzul','tagVerm','ResultadoVerm','OuroVerm','killsVerm','TorresVerm','InibidoresVerm','DragoesVerm','BaroesVerm','ArautosVerm','TimeVencedor'])

'''
As colunas do dataframe são criadas e preenchidas com os dados advindos do dataset auxiliar
Para as informações de Gold (Ouro), é utilizada a função apply(max) para que seja obtido
o valor máximo de dinheiro que aquele determinado time alcançou naquela partida
Para a quantidade de kills, torres, inibidores, dragões, barões e arautos, é utilizada a função apply(len)
para obter a quantidade de ocorrências de cada coluna.
Ou seja, se há 6 registros de torres destruídas naquela linha, então 6 torres foram destruídas no total.
Além disso, para as informações de Gold, kills, torres, inibidores, dragões, barões e arautos foi utilizada a função apply(eval) 
para converter  o tipo, pois cada linha dessa coluna contém uma list, porém essas lists vieram entre aspas, ou seja, em um tipo string.
'''
partidas.tagAzul = dsAux.blueTeamTag
partidas.ResultadoAzul = dsAux.bResult
partidas.OuroAzul = (dsAux.goldblue.apply(eval)).apply(max)
partidas.killsAzul = (dsAux.bKills.apply(eval)).apply(len)
partidas.TorresAzul = (dsAux.bTowers.apply(eval)).apply(len)
partidas.InibidoresAzul = (dsAux.bInhibs.apply(eval)).apply(len)
partidas.DragoesAzul =(dsAux.bDragons.apply(eval)).apply(len)
partidas.BaroesAzul = (dsAux.bBarons.apply(eval)).apply(len)
partidas.ArautosAzul = (dsAux.bHeralds.apply(eval)).apply(len)
'''
O mesmo ocorre para o lado vermelho
'''
partidas.tagVerm = dsAux.redTeamTag
partidas.ResultadoVerm = dsAux.rResult
partidas.OuroVerm = (dsAux.goldred.apply(eval)).apply(max)
partidas.killsVerm = (dsAux.rKills.apply(eval)).apply(len)
partidas.TorresVerm = (dsAux.rTowers.apply(eval)).apply(len)
partidas.InibidoresVerm = (dsAux.rInhibs.apply(eval)).apply(len)
partidas.DragoesVerm = (dsAux.rDragons.apply(eval)).apply(len)
partidas.BaroesVerm = (dsAux.rBarons.apply(eval)).apply(len)
partidas.ArautosVerm = (dsAux.rHeralds.apply(eval)).apply(len)

'''
Aqui filtramos o dataframe, restringindo apenas aos jogos da equipe brasileira PAIN Gaming
'''
partidas = partidas[(partidas.tagAzul == 'PNG') | (partidas.tagVerm == 'PNG')]

'''
Para saber qual time daquela partida venceu (do lado azul ou do lado vermelho), utilizou-se a função
np.where e, caso a coluna 'resultAzul' for igual a 1, significa que o lado azul venceu o jogo, então
a coluna timeVencedor recebe 1. Caso contrário, o time vermelho que venceu, então a coluna recebe 2.
'''
partidas.TimeVencedor = np.where(partidas.ResultadoAzul == 1, 1, 2)

'''
Divisão de Treino e Teste com as colunas que influenciam no resultado de um jogo.
(Quantidade de Gold; kills; torres e inibidores destruídos; dragões, barões e arautos abatidos).
O treino foi realizado em uma proporção de 7:3.
'''

X_train, X_test, y_train, y_test = train_test_split(partidas[['OuroAzul','killsAzul','TorresAzul','InibidoresAzul','DragoesAzul','BaroesAzul','ArautosAzul',
                                                              'OuroVerm','killsVerm','TorresVerm','InibidoresVerm','DragoesVerm','BaroesVerm','ArautosVerm']],
                                                    partidas['TimeVencedor'],test_size=0.30,random_state=101)
'''
Os códigos abaixo aplicam o GridSearch para analisar quais os parâmetros mais eficientes do SVM
quando se analisa os dados do dataframe de partidas. Os mesmos foram comentados pois levam certo tempo
para serem concluídos. Ao fim da análise, o valor de cada parâmetro adquirido foi incluído na fase
de treino do SVC
'''
# Aplicando o teste de parâmetros
# params_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear', 'sigmoid'], 'degree': [1, 3, 5, 7, 10]}
# grid = GridSearchCV(SVC(), params_grid, refit=True, verbose=3)
# grid.fit(x_train, y_train)
# print(grid.best_estimator_)

'''
Os parâmetros de melhor estimativa foram C=0.1, degree=1, gamma=1, kernel='linear'
Sendo assim, criou-se o modelo SVC e o mesmo foi treinado em seguida
'''
modelo = SVC(C=0.1, gamma=1, kernel='linear', degree=1, random_state=10)
modelo.fit(X_train, y_train)

'''
Deu-se início as predições dos resultados alcançados no treino, após finalizadas,
ocorreu o print do 'classification report'
'''
prediction = modelo.predict(X_test)
print(classification_report(y_test, prediction, zero_division=0))

'''
Por último, a matriz de confusão foi plotada em forma gráfica para melhor visualização nas linhas abaixo
'''
ax = sns.heatmap(confusion_matrix(y_test, prediction), annot=True, fmt='d',xticklabels=['Time Vermelho','Time Azul'], yticklabels=['Time Vermelho','Time Azul'])
ax.set(xlabel = 'True Label', ylabel='Predicted')
plt.show()