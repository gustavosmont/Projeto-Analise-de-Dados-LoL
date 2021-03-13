import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from ast import literal_eval

lol = "..\\datasetLoL\\LeagueofLegends.csv"
dataset = pd.read_csv(lol)

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')

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

partidas = pd.DataFrame()

partidas['tagAzul'] = dsAux['blueTeamTag']
partidas['resultAzul'] = dsAux['bResult']
partidas['goldAzul'] = dsAux['goldblue'].apply(max)
partidas['killsAzul'] = dsAux['bKills'].apply(len)
partidas['towersAzul'] = dsAux['bTowers'].apply(len)
partidas['inibsAzul'] = dsAux['bInhibs'].apply(len)
partidas['dragonsAzul'] = dsAux['bDragons'].apply(len)
partidas['baronsAzul'] = dsAux['bBarons'].apply(len)
partidas['heraldsAzul'] = dsAux['bHeralds'].apply(len)

partidas['tagVerm'] = dsAux['redTeamTag']
partidas['resultVerm'] = dsAux['rResult']
partidas['goldVerm'] = dsAux['goldred'].apply(max)
partidas['killsVerm'] = dsAux['rKills'].apply(len)
partidas['towersVerm'] = dsAux['rTowers'].apply(len)
partidas['inibsVerm'] = dsAux['rInhibs'].apply(len)
partidas['dragonsVerm'] = dsAux['rDragons'].apply(len)
partidas['baronsVerm'] = dsAux['rBarons'].apply(len)
partidas['heraldsVerm'] = dsAux['rHeralds'].apply(len)

partidas = partidas[(partidas.tagAzul == 'PNG') | (partidas.tagVerm == 'PNG')]
partidas = partidas.reset_index(drop=True)
partidas['timeVencedor'] = np.where(partidas.resultAzul == 1, 1, 2)

# print(partidas)

# Divisão de Treino e Teste
x = partidas[['goldAzul', 'killsAzul', 'towersAzul', 'inibsAzul', 'dragonsAzul', 'baronsAzul', 'heraldsAzul',
            'goldVerm', 'killsVerm', 'towersVerm', 'inibsVerm', 'dragonsVerm', 'baronsVerm', 'heraldsVerm']]
y = partidas['timeVencedor']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

# Treino do SVC
modelo = SVC(C=0.1, gamma=1, kernel='linear', degree=1, random_state=10)
modelo.fit(x_train, y_train)

# Aplicando o teste de parâmetros
# params_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear', 'sigmoid'], 'degree': [1, 3, 5, 7, 10]}
# grid = GridSearchCV(SVC(), params_grid, refit=True, verbose=3)
# grid.fit(x_train, y_train)
# print(grid.best_estimator_)
# Os parâmetros mais eficientes são SVC(C=0.1, degree=1, gamma=1, kernel='linear')

# Predições
prediction = modelo.predict(x_test)
print(classification_report(y_test, prediction, zero_division=0))