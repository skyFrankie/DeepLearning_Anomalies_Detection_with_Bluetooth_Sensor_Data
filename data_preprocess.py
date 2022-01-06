from config import*
import pandas as pd


def read_raw_csv(year,month,date):
    col_name = ['id', 'mac_id', 'node_id', 'date', 'time', 'duration']
    month_dict = {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'}
    temp = pd.read_csv(RAW_DATA_PATH/f'{year}_{month}_BTdata'/f'Data{month_dict[month]}{date}.csv',names=col_name)
    return temp

test = read_raw_csv(2017,11,14)
print(h3i)



