#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.cluster import KMeans
import pickle
from pulp import LpVariable, LpProblem, lpSum, LpMaximize, LpStatus, PULP_CBC_CMD
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class ClusterPredictor:
    def __init__(self, player_data, team_data, exp_team_data, cluster_model_path=None):
        self.player_data = player_data
        self.team_data = team_data
        self.exp_team_data = exp_team_data
        self.cluster_model = KMeans
        self.final_dataframe = self._transform_data(player_data, team_data, exp_team_data)
        
    def cal_points(self,z):
        z.rename(columns={'date': 'match_date'}, inplace=True)
        z['position'] = z['position'].str.strip().str.lower().replace("striker", "forward")
        z['match_date'] = pd.to_datetime(z['match_date'], format="%d-%m-%Y")
        z = z.sort_values(by=['player_id', 'match_date'], ascending=(True, True)).drop_duplicates().reset_index(drop=True)
        z = z[~z.duplicated(['player_id', 'game_id'])].reset_index(drop=True).fillna(0)
        z.drop('updated_at', axis=1, inplace=True)
        # z = z.groupby(['player_id']).nth(-1).reset_index()
        # conn.close()
        z['shots_in_box'] = z[['att_ibox_blocked',
                                        'att_ibox_miss',
                                        'att_ibox_post',
                                        'att_ibox_target',]].sum(axis=1)

        z['penalty_miss'] = z[['att_pen_miss','att_pen_post','att_pen_target']].fillna(0).sum(axis=1)                                             

        z.rename(columns={'ontarget_scoring_att':'shots_on_target','total_att_assist':'chance_created',
                                'total_scoring_att':'total_shots'},inplace=True)

        z.drop(['att_ibox_blocked',
                    'att_ibox_miss',
                    'att_ibox_post',
                    'att_ibox_target','att_pen_miss','att_pen_target'],axis=1,inplace=True)
        z['shots_on_target_acc'] = round(z['shots_on_target']/z['total_shots'],4)
        z['tackle_accuracy'] = round(z['won_tackle']/z['total_tackle'],4)
        z['passing_accuracy'] = round(z['accurate_pass']/z['total_pass'],4)
        z["goal_assist"] = z["goal_assist"]+z["assist_pass_lost"]+z["assist_blocked_shot"]+z["assist_attempt_saved"]+z["assist_post"]+z["assist_free_kick_won"]+z["assist_handball_won"]+z["assist_own_goal"]+z["assist_penalty_won"]
        points_dict = { 
                        'shots_on_target':6,
                        'chance_created':3,
                        'every_5_pass_completed':1,
                        'goal_by_gk_or_def': 60,
                        'goal_by_mid': 50,
                        'goal_by_fwd': 40,
                        'goal_assist': 20,
                        'won_tackle':4,
                        'interception':4,
                        'saves':6,
                        'clean_sheet_by_gk_or_def': 20,
                        'penalty_save': 50,
                        'penalty_miss': -20,
                        'goals_conceded_by_gk_or_def': -2,
                        'yellow_card': -4,
                        'red_card': -10,
                        'own_goals': -8,
                        'game_started': 4,
                    } 
        points_calculate_cols = list(points_dict.keys())
        z['goal_by_gk_or_def'] =  z.apply(lambda x : x.goals if (x.position in ['goalkeeper','defender']) else 0 ,axis=1)
        z['goal_by_mid'] =  z.apply(lambda x : x.goals if x.position == 'midfielder' else 0 ,axis=1)
        z['goal_by_fwd'] =  z.apply(lambda x : x.goals if x.position == 'forward' else 0 ,axis=1)
        z['clean_sheet_by_gk_or_def'] = z.apply(lambda x : x.clean_sheet if (x.position in ['goalkeeper','defender'] and x.mins_played >=54) else 0 ,axis=1)
        z['every_5_pass_completed'] = z.apply(lambda x : int(x.accurate_pass//5) if pd.isna(x.saves)!=True else 0 ,axis=1)
        z['goals_conceded_by_gk_or_def'] = z.apply(lambda x : int(x.goals_conceded) if (pd.isna(x.goals_conceded)!=True and x.position in ['goalkeeper','defender']) else 0,axis=1)
        z['total_points'] = z.apply(lambda x : sum([x[col]*points_dict[col] if pd.isna(x[col])!=True else 0 for col in points_calculate_cols]),axis=1)

        return z
    
    def transform_data(self,players_df,optimize_on="total_points"):
        df = players_df.copy()
        # replace original team abbv with 1 & 2
        teamNameMap = {i:j+1 for j, i in enumerate(df['team_id'].unique())}
        df['team'] = df['team_id'].replace(teamNameMap)
        # think it as availability, if player is injured/not available, set it to zero
        df['quantity'] = 1
        # onehot encode => player_role, teamName
        df = pd.get_dummies(df, columns=['team', 'position'])
        # defined feature columns
        feature_cols = [optimize_on, 'quantity', 
                        'team_1', 'team_2', 
                        'position_defender', 'position_forward',
                        'position_goalkeeper', 'position_midfielder']
        
        # check if all features are present
        for col in feature_cols:
            if col not in df.columns:
                if "team_1" in col:
                    error_msg = "Home team data missing to form Team"
                elif "team_2" in col:
                    error_msg = "Away team data missing to form Team"
                else:
                    error_msg = "%s data missing to form Team "%col.split("_")[-1]
                # self.send_error(self.league_id, match, error_msg)
                print("Required columns missing to form Team:", col)

        # Creates a list of the Players
        player_names = list(df['player_id'])
        feat_dict = {}
        for col in feature_cols:
            feat_dict[col] = dict(zip(player_names, df[col].values))
        return player_names, feat_dict

    def optimize_team(self,player_names, feat,optimize_on="total_points"):
        feature_cols = [optimize_on, 'quantity', 
                        'team_1', 'team_2', 
                        'position_defender', 'position_forward',
                        'position_goalkeeper', 'position_midfielder']
        # Players chosen 
        player_chosen = LpVariable.dicts("playerChosen", player_names, 0, 1, cat='Integer')
        # define np problem
        prob = LpProblem("Fantasy_Football", LpMaximize)
        # The objective function is added to 'prob' first
        prob += lpSum([feat[optimize_on][i]*player_chosen[i] for i in player_names]), "total_points"
        # Total
        prob += lpSum([feat['quantity'][f] * player_chosen[f] for f in player_names]) == 11, "Totalselection"
        # GK
        prob += lpSum([feat['position_goalkeeper'][f] * player_chosen[f] for f in player_names]) == 1, "GK"
        # DF
        prob += lpSum([feat['position_defender'][f] * player_chosen[f] for f in player_names]) >= 3, "DFmin"
        prob += lpSum([feat['position_defender'][f] * player_chosen[f] for f in player_names]) <= 5, "DFmax"
        # MF
        prob += lpSum([feat['position_midfielder'][f] * player_chosen[f] for f in player_names]) >= 3, "MFmin"
        prob += lpSum([feat['position_midfielder'][f] * player_chosen[f] for f in player_names]) <= 5, "MFmax"
        # FW
        prob += lpSum([feat['position_forward'][f] * player_chosen[f] for f in player_names]) >= 1, "FWmin"
        prob += lpSum([feat['position_forward'][f] * player_chosen[f] for f in player_names]) <= 3, "FWmax"
        # Team1
        prob += lpSum([feat['team_1'][f] * player_chosen[f] for f in player_names]) >= 4, "Team1Minimum"
        prob += lpSum([feat['team_1'][f] * player_chosen[f] for f in player_names]) <= 7, "Team1Maximum"
        # Team2
        prob += lpSum([feat['team_2'][f] * player_chosen[f] for f in player_names]) >= 4, "Team2Minimum"
        prob += lpSum([feat['team_2'][f] * player_chosen[f] for f in player_names]) <= 7, "Team2Maximum"
        prob.solve(PULP_CBC_CMD(msg=0))
#         print("Status:", LpStatus[prob.status])
        # print("points maximized = {}".format(round(value(prob.objective),2)))
        return prob

    def custom_condition_win(row):
        if row['winner_id'] == row['team_id']:
            return 1

    def custom_condition_draw(row):
        if row['winner_id'] == 0:
            return 1


    def _transform_data(self, player_data, team_data, exp_team_data):
        all_game_id_list = []
        
        pdf = player_data.copy()
        l_id = pdf['league_id'].unique()[0]
       
        for game_id in pdf['game_id'].unique().tolist():
            z = pdf[pdf.game_id==game_id]
            z = self.cal_points(z)
            player_names, feat = self.transform_data(z, optimize_on="total_points")
            prob = self.optimize_team(player_names, feat,optimize_on="total_points")
            optimal_squad = []
            for v in prob.variables():
                if v.varValue>0:
                    optimal_squad.append(v.name.split('_')[-1])
            z["in_optimal_11"] = z["player_id"].apply(lambda x : 1 if x in optimal_squad else 0)

            opt_df = z[z.in_optimal_11==1]
            opt_df['Home_Away'] = opt_df.apply(lambda row: '1-0' if row['home_id'] == row['team_id'] else '0-1', axis=1)
            opt_df['Pos_for']= opt_df.apply(lambda row:
                                        '0-0-0-1' if row['position'] == 'forward'
                                        else '0-1-0-0' if row['position'] == 'defender'
                                        else '0-0-1-0' if row['position'] == 'midfielder' 
                                        else '1-0-0-0' ,axis = 1  )
            cap = opt_df.sort_values(by='total_points',ascending=(False)).iloc[0]
            v_cap = opt_df.sort_values(by='total_points',ascending=(False)).iloc[1]
            cap_team = cap['Home_Away']
            cap_pos = cap['Pos_for']
            v_cap_team = v_cap['Home_Away']
            v_cap_pos = v_cap['Pos_for']
            opt_gk = opt_df.loc[opt_df['position'] == 'goalkeeper']['Home_Away'].values[0]


            tdef = opt_df[opt_df.position=="defender"].shape[0]
            tmef = opt_df[opt_df.position=="midfielder"].shape[0]
            tfw = opt_df[opt_df.position=="forward"].shape[0]
            team_H_ply = opt_df[opt_df.team_id==opt_df.home_id].shape[0]
            team_A_ply = opt_df[opt_df.team_id==opt_df.away_team_id].shape[0]
            tm_H_def = opt_df[(opt_df.position=="defender")&(opt_df.team_id==opt_df.home_id)].shape[0]
            tm_A_def = opt_df[(opt_df.position=="defender")&(opt_df.team_id==opt_df.away_team_id)].shape[0]
            tm_H_mf = opt_df[(opt_df.position=="midfielder")&(opt_df.team_id==opt_df.home_id)].shape[0]
            tm_A_mf = opt_df[(opt_df.position=="midfielder")&(opt_df.team_id==opt_df.away_team_id)].shape[0]
            tm_H_fw = opt_df[(opt_df.position=="forward")&(opt_df.team_id==opt_df.home_id)].shape[0]
            tm_A_fw = opt_df[(opt_df.position=="forward")&(opt_df.team_id==opt_df.away_team_id)].shape[0]
            tm_H_gk = opt_df[(opt_df.position=="goalkeeper")&(opt_df.team_id==opt_df.home_id)].shape[0]
            tm_A_gk = opt_df[(opt_df.position=="goalkeeper")&(opt_df.team_id==opt_df.away_team_id)].shape[0]

            dd = {   "game_id": game_id,
                     "league_id": l_id,
                    "formation" : str(tdef)+"-"+str(tmef)+"-"+str(tfw),
                    "team_ratio" : str(team_H_ply)+":"+str(team_A_ply),

                    "cap_team":cap_team,
                    "cap_position":cap_pos,
                    "v_cap_team":v_cap_team,
                    "v_cap_position":v_cap_pos,
                    "gk_position":opt_gk,

                    "def_ratio": str(tm_H_def)+":"+str(tm_A_def),
                    "mid_ratio": str(tm_H_mf)+":"+str(tm_A_mf),
                    "for_ratio": str(tm_H_fw)+":"+str(tm_A_fw),
                    


                }
            all_game_id_list.append(dd)
            
        final_dataframe = pd.DataFrame(all_game_id_list)
        id_to_team_id_mapping = team_data.set_index('game_id')['team_id'].to_dict()
        id_to_home_mapping = team_data.set_index('game_id')['home_id'].to_dict()
        id_to_away_mapping = team_data.set_index('game_id')['away_team_id'].to_dict()
        final_dataframe['team_id'] = final_dataframe['game_id'].map(id_to_team_id_mapping)

        final_dataframe['home_id'] = final_dataframe['game_id'].map(id_to_home_mapping)
        final_dataframe['away_team_id'] = final_dataframe['game_id'].map(id_to_away_mapping)
        td = team_data.copy()
        td =  td.fillna(0)
        td['GA'] = td['goals'] + td['goal_assist']
        td = td.sort_values(['team_id','date'],ascending=(True))
        td['Win'] = td.apply(lambda row: 1 if row['winner_id'] == row['team_id'] else 0, axis=1)
        td = td[['season','league','date','home_id','away_team_id','home_game','game_id','team_id','goals','goals_conceded','ontarget_scoring_att','total_att_assist','possession_percentage','Win','goal_assist','GA']]
        home_game = td[td['home_game'] == 1].reset_index(drop=True)
        away_game = td[td['home_game'] == 0].reset_index(drop=True)
   
 #        exp_team = exp_team_data.merge(td,on=["game_id","team_id"],how='inner')
#         exp_team = exp_team[['season','league','date','home_id','away_team_id','home_game','game_id','team_id','goals','goals_conceded',ontarget_scoring_att,total_att_assist,possession_percentage,won_tackles,interceptions,'Win','Draw']]
#         exp_team = exp_team.fillna(0)
#         exp_team = exp_team.sort_values(['team_id','date'],ascending=(True))
#         exp_team["xG"] = exp_team.groupby(['team_id'])['goals'].apply(lambda x : x.expanding().mean().shift(1))
#         exp_team["xGC"] = exp_team.groupby(['team_id'])['goals_conceded'].apply(lambda x : x.expanding().mean().shift(1))
#         exp_team["xWin"] = exp_team.groupby(['team_id'])['Win'].apply(lambda x : x.expanding().mean().shift(1))
#         exp_team["xDraw"] = exp_team.groupby(['team_id'])['Draw'].apply(lambda x : x.expanding().mean().shift(1))
#         exp_team['xG'] = exp_team['xG'].fillna(exp_team['goals'])
#         exp_team['xGC'] = exp_team['xGC'].fillna(exp_team['goals_conceded'])
#         exp_team['xWin'] = exp_team['xWin'].fillna(exp_team['Win'])
#         exp_team['xDraw'] = exp_team['xDraw'].fillna(exp_team['Draw'])

        #ontarget_scoring_att,total_att_assist,possession_percentage,won_tackles,interceptions
#         #mean, #std
#         home_game["xG_h"] = home_game.groupby(['team_id'])['goals'].apply(lambda x : x.expanding().mean().shift(1))
#         home_game["xG_h_sd"] = home_game.groupby(['team_id'])['goals'].apply(lambda x : x.expanding().std().shift(1))
        home_game["xGA_h"] = home_game.groupby(['team_id'])['GA'].apply(lambda x : x.expanding().mean().shift(1))
        home_game['xGA_h'] = home_game['xGA_h'].fillna(home_game['GA'])
        home_game["xGA_h_sd"] = home_game.groupby(['team_id'])['GA'].apply(lambda x : x.expanding().std().shift(1))
        #home_game = home_game.dropna(axis=0, subset=['xGA_h_sd'])
        
        
        
        home_game["xGC_h"] = home_game.groupby(['team_id'])['goals_conceded'].apply(lambda x : x.expanding().mean().shift(1))
        home_game['xGC_h'] = home_game['xGC_h'].fillna(home_game['goals_conceded'])
        home_game["xGC_h_sd"] = home_game.groupby(['team_id'])['goals_conceded'].apply(lambda x : x.expanding().std().shift(1))
        #home_game = home_game.dropna(axis=0, subset=['xGC_h_sd'])
        
        
        
#         home_game['xosa_h'] = home_game.groupby(['team_id'])['ontarget_scoring_att'].apply(lambda x : x.expanding().mean().shift(1))
#         home_game['xosa_h_sd'] = home_game.groupby(['team_id'])['ontarget_scoring_att'].apply(lambda x : x.expanding().std().shift(1))
#         home_game['xosa_h'] = home_game['xosa_h'].fillna(home_game['ontarget_scoring_att'])
#         home_game["xtaa_h"] = home_game.groupby(['team_id'])['total_att_assist'].apply(lambda x : x.expanding().mean().shift(1))
#         home_game["xtaa_h_sd"] = home_game.groupby(['team_id'])['total_att_assist'].apply(lambda x : x.expanding().std().shift(1))
#         home_game['xtaa_h'] = home_game['xtaa_h'].fillna(home_game['total_att_assist'])
#         home_game["xpp_h"] = home_game.groupby(['team_id'])['possession_percentage'].apply(lambda x : x.expanding().mean().shift(1))
#         home_game["xpp_h_sd"] = home_game.groupby(['team_id'])['possession_percentage'].apply(lambda x : x.expanding().std().shift(1))
#         home_game['xpp_h'] = home_game['xpp_h'].fillna(home_game['possession_percentage'])
        home_game["xWin_h"] = home_game.groupby(['team_id'])['Win'].apply(lambda x : x.expanding().mean().shift(1))
        home_game['xWin_h'] = home_game['xWin_h'].fillna(home_game['Win'])
        home_game["xWin_h_sd"] = home_game.groupby(['team_id'])['Win'].apply(lambda x : x.expanding().std().shift(1))
        #home_game = home_game.dropna(axis=0, subset=['xWin_h_sd'])
        home_game = home_game.dropna()
        
        home_game['dGA_h'] = np.where(
        home_game["xGA_h_sd"] != 0,
        home_game["xGA_h"] / home_game["xGA_h_sd"],
        0 )
        home_game['dGC_h'] = np.where(
        home_game["xGC_h_sd"] != 0,
        home_game["xGC_h"] / home_game["xGC_h_sd"],
        0 )
        home_game['dWin_h'] = np.where(
        home_game["xWin_h_sd"] != 0,
        home_game["xWin_h"] / home_game["xWin_h_sd"],
        0 )
        #home_game = home_game.fillna(0)
        home_cluster = home_game[['dGC_h','dGA_h',"dWin_h"]]
        scaler = StandardScaler()
        data_standardized_h = scaler.fit_transform(home_cluster)
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(data_standardized_h)

        self.cluster_model = kmeans

        cluster_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        home_game['cluster_h'] = cluster_labels
        
#         away_game["xG_a"] = away_game.groupby(['team_id'])['goals'].apply(lambda x : x.expanding().mean().shift(1))
#         away_game["xG_a_sd"] = away_game.groupby(['team_id'])['goals'].apply(lambda x : x.expanding().std().shift(1))
        away_game["xGA_a"] = away_game.groupby(['team_id'])['GA'].apply(lambda x : x.expanding().mean().shift(1))
        away_game['xGA_a'] = away_game['xGA_a'].fillna(away_game['GA'])
        away_game["xGA_a_sd"] = away_game.groupby(['team_id'])['GA'].apply(lambda x : x.expanding().std().shift(1))
        #away_game = away_game.dropna(axis=0, subset=['xGA_a_sd'])
        



        
        away_game["xGC_a"] = away_game.groupby(['team_id'])['goals_conceded'].apply(lambda x : x.expanding().mean().shift(1))
        away_game['xGC_a'] = away_game['xGC_a'].fillna(away_game['goals_conceded'])
        away_game["xGC_a_sd"] = away_game.groupby(['team_id'])['goals_conceded'].apply(lambda x : x.expanding().std().shift(1))
        #away_game = away_game.dropna(axis=0, subset=['xGC_a_sd'])
      
        
       
       # away_game['xosa_a'] = away_game.groupby(['team_id'])['ontarget_scoring_att'].apply(lambda x : x.expanding().mean().shift(1))
#         away_game['xosa_a_sd'] = away_game.groupby(['team_id'])['ontarget_scoring_att'].apply(lambda x : x.expanding().std().shift(1))
#         away_game['xosa_a'] = away_game['xosa_a'].fillna(away_game['ontarget_scoring_att'])
#         away_game["xtaa_a"] = away_game.groupby(['team_id'])['total_att_assist'].apply(lambda x : x.expanding().mean().shift(1))
#         away_game["xtaa_a_sd"] = away_game.groupby(['team_id'])['total_att_assist'].apply(lambda x : x.expanding().std().shift(1))
#         away_game['xtaa_a'] = away_game['xtaa_a'].fillna(away_game['total_att_assist'])
#         away_game["xpp_a"] = away_game.groupby(['team_id'])['possession_percentage'].apply(lambda x : x.expanding().mean().shift(1))
#         away_game["xpp_a_sd"] = away_game.groupby(['team_id'])['possession_percentage'].apply(lambda x : x.expanding().std().shift(1))
#         away_game['xpp_a'] = away_game['xpp_a'].fillna(away_game['possession_percentage'])
        away_game["xWin_a"] = away_game.groupby(['team_id'])['Win'].apply(lambda x : x.expanding().mean().shift(1))
        away_game['xWin_a'] = away_game['xWin_a'].fillna(away_game['Win'])
        away_game["xWin_a_sd"] = away_game.groupby(['team_id'])['Win'].apply(lambda x : x.expanding().std().shift(1))
        #away_game = away_game.dropna(axis=0, subset=['xWin_a_sd'])
        away_game = away_game.dropna()
        #away_game['dGA_a'] = away_game["xGA_a"]/away_game["xGA_a_sd"]
        away_game['dGA_a'] = np.where(
        away_game["xGA_a_sd"] != 0,
        away_game["xGA_a"] / away_game["xGA_a_sd"],
        0 )
        away_game['dGC_a'] = np.where(
        away_game["xGC_a_sd"] != 0,
        away_game["xGC_a"] / away_game["xGC_a_sd"],
        0 )
        away_game['dWin_a'] = np.where(
        away_game["xWin_a_sd"] != 0,
        away_game["xWin_a"] / away_game["xWin_a_sd"],
        0 )
        away_cluster = away_game[['dGC_a','dGA_a',"dWin_a"]]
        scaler = StandardScaler()
        data_standardized_a = scaler.fit_transform(away_cluster)
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(data_standardized_a)

        self.cluster_model = kmeans

        cluster_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        away_game['cluster_a'] = cluster_labels
        
        exp_team = home_game.merge(away_game,on = ['game_id'],how='inner')
#         exp_team = home_away_df[["game_id","team_id","xG","xGC","xGsd","xGCsd"]].reset_index(drop=True)
        final_dataframe = final_dataframe.merge(exp_team, on=['game_id'],how="inner")
#         final_dataframe = final_dataframe.merge(exp_team, left_on=['game_id','away_team_id'],right_on=['game_id','team_id'],how="inner")


    
#         X_c = final_dataframe[["xG_h","xGC_h",'xGC_a','xG_a','xosa_h','xosa_a',"xtaa_h","xtaa_a","xpp_h","xpp_a","xG_h_sd","xGC_h_sd",'xosa_h_sd',"xtaa_h_sd",'xpp_h_sd',"xG_a_sd","xGC_a_sd",'xosa_a_sd',"xtaa_a_sd",'xpp_a_sd','xWin_a','xWin_h']]
# #         final_dataframe['xGhome'] = final_dataframe['xG_x']*0.6 + final_dataframe['xGC_y']*0.4
# #         final_dataframe['xGaway'] = final_dataframe['xG_y']*0.5 + final_dataframe['xGC_x']*0.5
# #         X_c = final_dataframe[["xGhome","xGaway","xWin_x","xWin_y"]]

#         kmeans = KMeans(n_clusters=4, random_state=42)
#         kmeans.fit(X_c)

#         self.cluster_model = kmeans

#         cluster_labels = kmeans.labels_
#         cluster_centers = kmeans.cluster_centers_
#         final_dataframe['cluster'] = cluster_labels
        
         
#         file_path = f'{l_id}_cluster_model.pickle'    
#         with open(file_path, 'wb') as f:
#             pickle.dump(self.cluster_model, f)
    
#         final_dataframe.to_excel(f'{l_id}_final_data.xlsx')
        return final_dataframe
        

    def get_cluster_for_game(self, game_id, home_id, away_team_id):
        game_data = self.final_dataframe[(self.final_dataframe['game_id'] == game_id) & (self.final_dataframe['home_id'] == home_id) & (self.final_dataframe['away_team_id'] == away_team_id)]
        if not game_data.empty:
            return game_data['cluster'].iloc[0]
        else:
            return None

   
    def get_cluster_centers(self):
        if self.cluster_model is None:
            raise ValueError("Cluster model has not been trained yet.")
        
        cluster_centers = self.cluster_model.cluster_centers_
        cluster_numbers = list(range(len(cluster_centers)))
        cluster_info = dict(zip(cluster_numbers, cluster_centers))
        return cluster_info

#     def create_nested_dict(self, result_dict, keys, value):
#         current_dict = result_dict
#         for key in keys[:-1]:
#             current_dict = current_dict.setdefault(key, {})
#         current_dict[keys[-1]] = value

#     def generate_optimal_team_summary(self):
#         result_dict = {}

#         # Find top team ratios for each cluster
#         grouped_data = self.final_dataframe.groupby(['cluster', 'team_ratio', 'formation', 'def_ratio', 'mid_ratio', 'for_ratio']).size().reset_index(name='count')

#         for cluster, cluster_data in grouped_data.groupby('cluster'):
#             team_ratio_counts = cluster_data.groupby('team_ratio')['count'].sum()
#             total_values = team_ratio_counts.sum()
#             threshold = total_values * 0.6
#             top_team_ratios = team_ratio_counts.nlargest(2)
#             if top_team_ratios.iloc[0] >= threshold:
#                 top_team_ratios = top_team_ratios.iloc[:1]
#             top_team_ratios = top_team_ratios.index.tolist()

#             for team_ratio in top_team_ratios:
#                 team_ratio_data = cluster_data[cluster_data['team_ratio'] == team_ratio]
#                 formation_counts = team_ratio_data.groupby('formation')['count'].sum()
#                 total_values = formation_counts.sum()
#                 threshold = total_values * 0.6
#                 top_formations = formation_counts.nlargest(2)
#                 if top_formations.iloc[0] >= threshold:
#                     top_formations = top_formations.iloc[:1]
#                 top_formations = top_formations.index.tolist()

#                 team_ratio_formations = {}
#                 for formation in top_formations:
#                     formation_data = team_ratio_data[team_ratio_data['formation'] == formation]
#                     def_ratio_counts = formation_data.groupby('def_ratio')['count'].agg(['sum', 'size'])
#                     mid_ratio_counts = formation_data.groupby('mid_ratio')['count'].agg(['sum', 'size'])
#                     for_ratio_counts = formation_data.groupby('for_ratio')['count'].agg(['sum', 'size'])

#                     top_def_ratios = def_ratio_counts.nlargest(2, 'sum')['sum'].index.tolist()
#                     top_mid_ratios = mid_ratio_counts.nlargest(2, 'sum')['sum'].index.tolist()
#                     top_for_ratios = for_ratio_counts.nlargest(2, 'sum')['sum'].index.tolist()

#                     formation_ratios = {
#                         'def_ratios': [{'ratio': ratio, 'count': count} for ratio, count in zip(top_def_ratios, def_ratio_counts.loc[top_def_ratios, 'sum'])],
#                         'mid_ratios': [{'ratio': ratio, 'count': count} for ratio, count in zip(top_mid_ratios, mid_ratio_counts.loc[top_mid_ratios, 'sum'])],
#                         'for_ratios': [{'ratio': ratio, 'count': count} for ratio, count in zip(top_for_ratios, for_ratio_counts.loc[top_for_ratios, 'sum'])]
#                     }

#                     self.create_nested_dict(team_ratio_formations, [formation], formation_ratios)

#                 self.create_nested_dict(result_dict, [cluster, 'team_ratio', team_ratio], team_ratio_formations)

#         return result_dict


