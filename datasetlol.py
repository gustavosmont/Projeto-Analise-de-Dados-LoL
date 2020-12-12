
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
file = "datasetLoL\\matchinfo.csv"
dataset = pd.read_csv(file)
CBLol2017 = dataset[(dataset.League == 'CBLoL') & (dataset.Year == 2017) & (dataset.Season == 'Summer')]
matchup=pd.DataFrame(columns=['Top','Jungle','Mid','ADC','Suporte','Time'])
matchup['Top']=np.concatenate((CBLol2017.redTopChamp,CBLol2017.blueTopChamp));
matchup['Jungle']=np.concatenate((CBLol2017.redJungleChamp,CBLol2017.blueJungleChamp));
matchup['Mid']=np.concatenate((CBLol2017.redMiddleChamp,CBLol2017.blueMiddleChamp));
matchup['ADC']=np.concatenate((CBLol2017.redADCChamp,CBLol2017.blueADCChamp));
matchup['Suporte']=np.concatenate((CBLol2017.redSupportChamp,CBLol2017.blueSupportChamp));
matchup['Time']=np.concatenate((CBLol2017.redTeamTag,CBLol2017.blueTeamTag));
print("Top:",(matchup[(matchup.Time == 'PNG')].Top.describe().top),"||","Jungle:",(matchup[(matchup.Time == 'PNG')].Jungle.describe().top),"||","Mid:",(matchup[(matchup.Time == 'PNG')].Mid.describe().top),"||","ADC:",(matchup[(matchup.Time == 'PNG')].ADC.describe().top),"||","Suporte:",(matchup[(matchup.Time == 'PNG')].Suporte.describe().top))
print((matchup[(matchup.Time == 'PNG')].describe()))




