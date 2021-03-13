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
from ast import literal_eval
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
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')
'''
É feito uma cópia do dataset principal para um auxiliar
'''
dsAux = dataset.copy(deep=True)

dsAux['goldred'] = dsAux['goldred'].apply(literal_eval)
dsAux['goldblue'] = dsAux['goldblue'].apply(literal_eval)
dsAux['rKills'] = dsAux['rKills'].apply(literal_eval)
dsAux['bKills'] = dsAux['bKills'].apply(literal_eval)
dsAux['rTowers'] = dsAux['rTowers'].apply(literal_eval)
dsAux['bTowers'] = dsAux['bTowers'].apply(literal_eval)
dsAux['rInhibs'] = dsAux['rInhibs'].apply(literal_eval)
dsAux['bInhibs'] = dsAux['bInhibs'].apply(literal_eval)
dsAux['rDragons'] = dsAux['rDragons'].apply(literal_eval)
dsAux['bDragons'] = dsAux['bDragons'].apply(literal_eval)
dsAux['rBarons'] = dsAux['rBarons'].apply(literal_eval)
dsAux['bBarons'] = dsAux['bBarons'].apply(literal_eval)
dsAux['rHeralds'] = dsAux['rHeralds'].apply(literal_eval)
dsAux['bHeralds'] = dsAux['bHeralds'].apply(literal_eval)
'''
O nosso dataframe principal é criado com o nome de 'partidas'
'''
partidas = pd.DataFrame()
'''
As colunas do dataframe são criadas e preenchidas com os dados advindos do dataset auxiliar
Para as informações de Gold (moedas), é utilizada a função apply(max) para que seja obtido
o valor máximo de dinheiro que aquele determinado time alcançou naquela partida
Para a quantidade de kills, torres, inibidores, dragões, barões e arautos, é utilizada a função apply(len)
para obter a quantidade de ocorrências de cada coluna.
Ou seja, se há 6 registros de torres destruídas naquela linha, então 6 torres foram destruídas no total
'''
partidas['tagAzul'] = dsAux['blueTeamTag']
partidas['resultAzul'] = dsAux['bResult']
partidas['goldAzul'] = dsAux['goldblue'].apply(max)
partidas['killsAzul'] = dsAux['bKills'].apply(len)
partidas['towersAzul'] = dsAux['bTowers'].apply(len)
partidas['inibsAzul'] = dsAux['bInhibs'].apply(len)
partidas['dragonsAzul'] = dsAux['bDragons'].apply(len)
partidas['baronsAzul'] = dsAux['bBarons'].apply(len)
partidas['heraldsAzul'] = dsAux['bHeralds'].apply(len)
'''
O mesmo ocorre para o lado vermelho
'''
partidas['tagVerm'] = dsAux['redTeamTag']
partidas['resultVerm'] = dsAux['rResult']
partidas['goldVerm'] = dsAux['goldred'].apply(max)
partidas['killsVerm'] = dsAux['rKills'].apply(len)
partidas['towersVerm'] = dsAux['rTowers'].apply(len)
partidas['inibsVerm'] = dsAux['rInhibs'].apply(len)
partidas['dragonsVerm'] = dsAux['rDragons'].apply(len)
partidas['baronsVerm'] = dsAux['rBarons'].apply(len)
partidas['heraldsVerm'] = dsAux['rHeralds'].apply(len)
'''
Aqui filtramos o dataframe, restringindo apenas aos jogos da equipe brasileira PAIN Gaming
'''
partidas = partidas[(partidas.tagAzul == 'PNG') | (partidas.tagVerm == 'PNG')]
partidas = partidas.reset_index(drop=True)
'''
Para saber qual time daquela partida venceu (do lado azul ou do lado vermelho), utilizou-se a função
np.where e, caso a coluna 'resultAzul' for igual a 1, significa que o lado azul venceu o jogo, então
a coluna timeVencedor recebe 1. Caso contrário, o time vermelho que venceu, então a coluna recebe 2.
'''
partidas['timeVencedor'] = np.where(partidas.resultAzul == 1, 1, 2)

'''
Divisão de Treino e Teste com as colunas que influenciam no resultado de um jogo.
(Quantidade de Gold; kills; torres e inibidores destruídos; dragões, barões e arautos abatidos)
'''
x = partidas[['goldAzul', 'killsAzul', 'towersAzul', 'inibsAzul', 'dragonsAzul', 'baronsAzul', 'heraldsAzul',
            'goldVerm', 'killsVerm', 'towersVerm', 'inibsVerm', 'dragonsVerm', 'baronsVerm', 'heraldsVerm']]
y = partidas['timeVencedor']
'''
O treino foi realizado em uma proporção de 7:3
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)
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
modelo.fit(x_train, y_train)
'''
Deu-se início as predições dos resultados alcançados no treino, após finalizadas,
ocorreu o print do 'classification report'
'''
prediction = modelo.predict(x_test)
print(classification_report(y_test, prediction, zero_division=0))
'''
Por último, a matriz de confusão foi plotada em forma gráfica para melhor visualização nas linhas abaixo
'''
ax = sns.heatmap(confusion_matrix(y_test, prediction), annot=True, fmt='d',xticklabels=['Red Team','Blue Team'], yticklabels=['Red Team','Blue Team'])
ax.set(xlabel = 'True Label', ylabel='Predicted')
plt.show()