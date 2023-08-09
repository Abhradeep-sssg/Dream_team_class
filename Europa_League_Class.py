#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from urllib.parse import quote
import requests


# In[2]:


def db_connect():
    db_host = "10.1.0.10:3306"
    db_user = "abhradeep.roy"
    db_pass = "YlyEnr3V4eU9L9!i"
    cnx = create_engine("mysql+pymysql://{}:{}@{}/football".format(quote(db_user),quote(db_pass),db_host))
    conn = cnx.connect()
    return conn,cnx
conn,cnx = db_connect()


# In[3]:


df = pd.read_sql("select * from Team_data ",conn)
df = df.drop_duplicates(subset=["team_id","game_id"]).reset_index(drop=True)
europa = df[df['league_id'] == '6'][['season','league','week','home','home_id','away_team_id','opponent','game_id','team_id','winner_id','goals_conceded']]


# In[4]:


game_ids = list(europa['game_id'])
ll = []
for game_id in game_ids:
    url = "https://omo.akamai.opta.net/?game_id={}&feed_type=F9&user=fankhada1&psw=fankdaha!1&json".format(game_id)
    r = requests.get(url)
    dic= r.json()
    dic['SoccerFeed']['SoccerDocument']
    if isinstance(dic['SoccerFeed']['SoccerDocument'],list):
        ll.append(dic['SoccerFeed']['SoccerDocument'][0]['Competition']['Round'])
    else:
        ll.append(dic['SoccerFeed']['SoccerDocument']['Competition']['Round'])


# In[5]:


round_df = pd.DataFrame(ll)
round_inf = pd.DataFrame(list(europa['game_id']),list(round_df['Name'])).reset_index()
round_inf['Pool'] = round_df['Pool']
round_inf.rename(columns = {'index':'Round',0:'game_id'}, inplace = True)
id_to_round_name_mapping = round_inf.set_index('game_id')['Round'].to_dict()
id_to_group_name_mapping = round_inf.set_index('game_id')['Pool'].to_dict()
europa['Round'] = europa['game_id'].map(id_to_round_name_mapping)
europa['Pool'] = europa['game_id'].map(id_to_group_name_mapping)


# In[ ]:


game_ids = list(europa['game_id'])
ll = []
for game_id in game_ids:
    url = "https://omo.akamai.opta.net/?game_id={}&feed_type=F9&user=fankhada1&psw=fankdaha!1&json".format(game_id)
    r = requests.get(url)
    dic= r.json()
    dic['SoccerFeed']['SoccerDocument']
    if isinstance(dic['SoccerFeed']['SoccerDocument'],list):
        ll.append(dic['SoccerFeed']['SoccerDocument'][0]['Competition']['Round'])
    else:
        ll.append(dic['SoccerFeed']['SoccerDocument']['Competition']['Round'])
        
round_df = pd.DataFrame(ll)
round_inf = pd.DataFrame(list(europa['game_id']),list(round_df['Name'])).reset_index()
round_inf['Pool'] = round_df['Pool']
round_inf.rename(columns = {'index':'Round',0:'game_id'}, inplace = True)
id_to_round_name_mapping = round_inf.set_index('game_id')['Round'].to_dict()
id_to_group_name_mapping = round_inf.set_index('game_id')['Pool'].to_dict()
europa['Round'] = europa['game_id'].map(id_to_round_name_mapping)
europa['Pool'] = europa['game_id'].map(id_to_group_name_mapping)


# In[24]:


import pandas as pd

class EuropaLeagueCalculator:
    def __init__(self, data):
        self.data = data

    def custom_condition_win(self, row):
        if (row['winner_id'] == row['team_id']) and (row["Round"] == 'Play-Offs'):
            return 1
        elif (row['winner_id'] == row['team_id']):
            return 2 

    def custom_condition_loss(self, row):
        if row['winner_id'] != row['team_id'] and row['winner_id'] != 0:
            return 1

    def custom_condition_draw(self, row):
        if (row['winner_id'] == 0) and (row["Round"] == 'Play-Offs'):
            return 0.5
        elif row['winner_id'] == 0:
            return 1

    def add_points(self, row):
        if (len(row['Round']) > 1) and ('Play-Offs' in row['Round']):
            return 4 + (len(row['Round']) - 2)
        elif (len(row['Round']) == 1) and ('Play-Offs' in row['Round']):
            return 0
        else:
            return 4 + (len(row['Round']) - 1)

    def group_points(self, row):
        if row['position'] == 1:
            return 4
        elif row['position'] == 2:
            return 2
        else:
            return 0

    def calculate_goal_difference(self, row):
        opponent_goals_conceded = self.data.loc[
            (self.data['game_id'] == row['game_id']) & 
            (self.data['team_id'] != row['team_id']),
            'goals_conceded'
        ].values[0]

        return opponent_goals_conceded - row['goals_conceded']

    def find_top_2_teams(self, group_data):
        sorted_group = group_data.sort_values(by=["points", "goal_difference"], ascending=False)

        if sorted_group["points"].iloc[0] == sorted_group["points"].iloc[1]:
            top_2_teams = sorted_group.head(2).sort_values(by="goal_difference", ascending=False)
        else:
            if sorted_group["points"].iloc[1] == sorted_group["points"].iloc[2]:
                top_2_teams = sorted_group.head(3).sort_values(by="goal_difference", ascending=False).head(2)
            else:
                top_2_teams = sorted_group.head(2)

        return top_2_teams

    def calculate(self):
        self.data['winner_id'] = self.data['winner_id'].fillna(0)
        self.data['goals_conceded'] = self.data['goals_conceded'].fillna(0)
        self.data['Win'] = self.data.apply(self.custom_condition_win, axis=1)
        self.data['Draw'] = self.data.apply(self.custom_condition_draw, axis=1)
        self.data['Loss'] = self.data.apply(self.custom_condition_loss, axis=1)

        europa_group = self.data[self.data['Round'] == 'Round']
        europa_group['goal_difference'] = europa_group.apply(self.calculate_goal_difference, axis=1)
        group_points = europa_group.groupby(['season','team_id']).agg({'Win': 'sum', 'Draw': 'sum','goal_difference':'sum','Pool':'first','game_id':'nunique'}).reset_index()
        group_points['points'] = group_points['Win'] + group_points['Draw']
        grouped_data = group_points.groupby(["Pool","season"])

        top_2_teams_per_group = pd.DataFrame()

        for group_name, group_data in grouped_data:
            top_2_teams = self.find_top_2_teams(group_data)
            top_2_teams["Position"] = range(1, len(top_2_teams) + 1)
            top_2_teams_per_group = pd.concat([top_2_teams_per_group, top_2_teams])

        position_mapping = dict(zip(top_2_teams_per_group['team_id'].astype(str) + top_2_teams_per_group['season'].astype(str) + top_2_teams_per_group['Pool'].astype(str), top_2_teams_per_group['Position']))
        self.data['position'] = self.data['team_id'].astype(str) + self.data['season'].astype(str) + self.data['Pool'].astype(str)
        self.data['position'] = self.data['position'].map(position_mapping)
        points_by_team = self.data.groupby(['team_id','season','league']).agg({'Win': 'sum', 'Draw': 'sum','Round': 'unique','position':'first'}).reset_index()
        points_by_team['Bonus_Points'] = points_by_team.apply(self.add_points, axis=1)
        points_by_team['Group_Bonus_Points'] = points_by_team.apply(self.group_points, axis=1)
        points_by_team['Total_points'] = points_by_team['Win'] + points_by_team['Draw'] + points_by_team['Bonus_Points']+points_by_team['Group_Bonus_Points']
        
        pivot_df = points_by_team.pivot_table(index='team_id', columns='season', values='Total_points', fill_value=0).reset_index()
        pivot_df['Total_Score'] = pivot_df['2020']+pivot_df['2021']+pivot_df['2022']
        return pivot_df


# In[28]:


europa_calculator = EuropaLeagueCalculator(europa)
result = europa_calculator.calculate()

id_to_team_name_mapping = df.set_index('team_id')['team_name'].to_dict()
result['Team_name'] = result['team_id'].map(id_to_team_name_mapping)
result.sort_values(['Total_Score'],ascending=(False)).reset_index(drop=True).head(20)


# In[ ]:




