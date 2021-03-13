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

print(partidas)