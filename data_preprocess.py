from config import*
import pandas as pd
import datetime
import numpy as np
from pyspark.sql import SparkSession
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

# corridor_M = [10128,10079,10078,10080,10086,10139,10282,10034,10182]
# corridor_C_right = [10345,10093,10167,10322,10175,10281,10210,10089,10264,10152,10278,10088,10364,10181,10203,10087,10188,10221]
# corridor_C_left = [10345,10093,10167,10322,10175,10214,10219,10222,10210,10089,10264,10152,10278,10088,10364,10181,10203,10087,10188,10221]
# corridor_C_left.reverse()

class Dataprocessor():
    def __init__(self,start_date,end_date):
        self.corridor_M = self.read_location_csv('M')
        self.corridor_C = self.read_location_csv('C')
        self.start_date = start_date
        self.end_date = end_date

    def read_location_csv(self,coridor_name):
        temp = pd.read_csv(LOCATION_DATA_PATH/f'Corridor_{coridor_name}.csv')
        temp = temp[['ExtendedData/Data/0/value','longitude','latitude','ExtendedData/Data/5/value','left','right']].rename({'ExtendedData/Data/0/value':'node_id','ExtendedData/Data/5/value':'street_near_by'},axis=1)
        return temp

    def identify_time_interval(self,targe_time, time_array):
        """
        Assistant function used in read_raw_travel_time_csv()
        :param targe_time: time value
        :param time_array: array that contains the series from 0 to 24, 5/60 each
        :return: corresponding time interval
        """
        temp_num = 24
        for i in time_array:
            if i < targe_time:
                continue
            else:
                temp_num = i
                break
        return temp_num

    def read_raw_travel_time_csv(self,target_date):
        """
        Read date by date data and filter out non-sense data with the following logic:
        1. Remove incorrect timestamp data, as the time should not exceed 24
        2. Remove unrelated node data
        3. Remove extreme duration time data, as a vehicle is not like to stuck on one spot for more the one hour.
           Those possibly are parking cars or people staying nearby. Therefore filterd out 'duration' > 3600
        Add column to indicate the  time interval of each data point
        :param target_date: Date Object. The date of target file
        :return: Return lists of processed dataframes, by directions of each corridor
        """
        col_name = ['id', 'mac_id', 'node_id', 'date', 'time', 'duration']
        file_path = TIME_DATA_PATH/f'{target_date.strftime("%Y")}_{str(int(target_date.strftime("%m")))}_BTdata'

        temp = pd.read_csv(file_path/f'Data{target_date.strftime("%b").lower()}{str(int(target_date.strftime("%d")))}.csv',names=col_name)
        temp['time_interval'] = temp['time'].apply(lambda x: self.identify_time_interval(x, np.arange(0,24,5/60)))

        temp_M = temp[(temp['node_id'].isin(self.corridor_M['node_id'].unique().tolist())) & (temp['time'] < 24) & (temp['duration'] <= 3600)]
        temp_M['node_id'] = temp_M['node_id'].astype(int)
        temp_M = temp_M.merge(self.corridor_M, how='left', on='node_id')
        self.corridor_M_left_order = self.corridor_M.sort_values(['left'])['node_id'].unique().tolist()
        self.corridor_M_right_order = self.corridor_M.sort_values(['right'])['node_id'].unique().tolist()
        corridor_M_left = [temp_M[temp_M['node_id']==i]for i in self.corridor_M_left_order]
        corridor_M_right = [temp_M[temp_M['node_id']==i]for i in self.corridor_M_right_order]

        temp_C = temp[(temp['node_id'].isin(self.corridor_C['node_id'].unique().tolist())) & (temp['time'] < 24) & (temp['duration'] <= 3600)]
        temp_C['node_id'] = temp_C['node_id'].astype(int)
        temp_C = temp_C.merge(self.corridor_C, how='left', on='node_id')
        self.corridor_C_left_order = self.corridor_C[self.corridor_C['left'].notnull()].sort_values(['left'])['node_id'].unique().tolist()
        self.corridor_C_right_order = self.corridor_C[self.corridor_C['right'].notnull()].sort_values(['right'])['node_id'].unique().tolist()
        corridor_C_left = [temp_C[temp_C['node_id']==i]for i in self.corridor_C_left_order]
        corridor_C_right = [temp_C[temp_C['node_id']==i]for i in self.corridor_C_right_order]

        return corridor_M_left,corridor_M_right,corridor_C_left,corridor_C_right

    def process_direction_data(self):
        """
        Process the list of dataframes created from read_raw_travel_time_csv
        Output csv with added up route travel time date by date.
        """
        start_date = self.start_date
        end_date = self.end_date
        delta = datetime.timedelta(days=1)
        spark = SparkSession.builder.getOrCreate()
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        time_interval_reference = pd.DataFrame({'time_interval_1': list(np.append(np.arange(0, 24, 5 / 60), [24]).reshape(-1))})
        directory_list = ['Corridor_M_left','Corridor_M_right','Corridor_C_left','Corridor_C_right']
        while start_date <= end_date:
            try:
                corridor_M_left, corridor_M_right, corridor_C_left, corridor_C_right = self.read_raw_travel_time_csv(start_date)
                for j, links in enumerate([corridor_M_left, corridor_M_right, corridor_C_left, corridor_C_right]):
                    links_add_up = []
                    OUTPUT_DIR_PATH = OUTPUT_DATA_PATH /directory_list[j]
                    if not os.path.isdir(OUTPUT_DIR_PATH):
                        os.mkdir(OUTPUT_DIR_PATH)

                    for i in range(0, len(links) - 1):
                        temp1 = links[i].copy()
                        temp1.columns = temp1.columns.map(lambda x: x + '_1').tolist()
                        links[i].to_csv(DATA_PATH/'temp_1.csv',index=False)
                        left_spark = spark.read.option('header',True).option('inferSchema',True).option('delimiter',',').csv(str(DATA_PATH/'temp_1.csv'))
                        temp2 = links[i+1].copy()
                        temp2.columns = temp2.columns.map(lambda x: x + '_2').tolist()
                        links[i+1].to_csv(DATA_PATH/'temp_2.csv', index=False)
                        right_spark = spark.read.option('header',True).option('inferSchema',True).option('delimiter',',').csv(str(DATA_PATH/'temp_2.csv'))
                        left_node_id, right_node_id = links[i]['node_id'].values[0],links[i+1]['node_id'].values[0]
                        link = left_spark.join(right_spark,[(left_spark .mac_id_1 == right_spark.mac_id_2),(left_spark.time_1 < right_spark.time_2)],how='inner')
                        link = link.toPandas()
                        link['Travel_time'] = link['time_2'] - link['time_1'] + link['duration_1'] / 3600
                        link = link[link['Travel_time'] <= 1].groupby('time_interval_1').agg(Travel_time=('Travel_time','mean')).reset_inedx()
                        link = time_interval_reference.merge(link,how='left',on='time_interval_1').fillna(link['Travel_time'].mean()).rename({'Travel_time':f'{left_node_id}_{right_node_id}'},axis=1).set_index('time_interval_1')
                        links_add_up.append(link)
                        print(f'Finished {i} out of {len(links)-1}')
                    print(f'Done {j}')
                    links_add_up = pd.concat(links_add_up,axis=1)
                    links_add_up['Total_travel_time'] = links_add_up.sum(axis=1)
                    links_add_up['Date'] = start_date.strftime('%D')
                    links_add_up['Day'] = start_date.strftime('%w')
                    links_add_up.to_csv(OUTPUT_DIR_PATH/f"{start_date.strftime('%D').replace('/','_')}.csv",index=False)
                start_date = start_date + delta
            except:
                start_date = start_date + delta
                continue
        spark.stop()
        del spark
        return None



start_date = datetime.date(2017,1,1)
end_date = datetime.date(2018,1,31)
a = Dataprocessor(start_date,end_date)
a.process_direction_data()



