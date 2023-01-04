from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ELO_CONSTANT = 40
HOME_ELO_ADVANTAGE = 75
#Inputs: Winner's current elo, losers current elo, whether or not the home team won
#Teams at home have a 100 boost to their elo
#Returns a tuple of (new winner elo, new loser elo)
def calculate_elo(winner_elo, loser_elo, home_win):
    if home_win:
        new_winner_elo = winner_elo + ELO_CONSTANT * (1 - (1/(1+10**((loser_elo-HOME_ELO_ADVANTAGE-winner_elo)/400))))
        new_loser_elo = loser_elo + ELO_CONSTANT * (0 - (1/(1+10**((winner_elo+HOME_ELO_ADVANTAGE-loser_elo)/400))))
    else:
        new_winner_elo = winner_elo + ELO_CONSTANT * (1 - (1/(1+10**((loser_elo+HOME_ELO_ADVANTAGE-winner_elo)/400))))
        new_loser_elo = loser_elo + ELO_CONSTANT * (0 - (1/(1+10**((winner_elo-HOME_ELO_ADVANTAGE-loser_elo)/400))))
    return (new_winner_elo,new_loser_elo)

def predict_outcome(home_elo, away_elo):
    return (home_elo + HOME_ELO_ADVANTAGE) > away_elo

DEFAULT_ELO = 1500
NBA_TEAMS = {
"Atlanta Hawks":"ATL",
"Brooklyn Nets":"BKN",
"Boston Celtics":"BOS",
"Charlotte Hornets":"CHA",
"Chicago Bulls":"CHI",
"Cleveland Cavaliers":"CLE",
"Dallas Mavericks":"DAL",
"Denver Nuggets":"DEN",
"Detroit Pistons":"DET",
"Golden State Warriors":"GSW",
"Houston Rockets":"HOU",
"Indiana Pacers":"IND",
"Los Angeles Clippers":"LAC",
"Los Angeles Lakers":"LAL",
"Memphis Grizzlies":"MEM",
"Miami Heat":"MIA",
"Milwaukee Bucks":"MIL",
"Minnesota Timberwolves":"MIN",
"New Orleans Pelicans":"NOP",
"New York Knicks":"NYK",
"Oklahoma City Thunder":"OKC",
"Orlando Magic":"ORL",
"Philadelphia 76ers":"PHI",
"Phoenix Suns":"PHX",
"Portland Trail Blazers":"POR",
"Sacramento Kings":"SAC",
"San Antonio Spurs":"SAS",
"Toronto Raptors":"TOR",
"Utah Jazz":"UTA",
"Washington Wizards":"WAS"
}

TEAM_ELO = {}
for key in NBA_TEAMS.keys():
    TEAM_ELO[NBA_TEAMS[key]] = DEFAULT_ELO

SEASON_MONTHS = ["october", "november", "december", "january"]
frames = []
total_entries = 0
for month in SEASON_MONTHS:
    url =  f"https://www.basketball-reference.com/leagues/NBA_2023_games-{month}.html"

    html = urlopen(url)
    soup = BeautifulSoup(html, features="lxml")

    headers = [th.getText() for th in soup.findAll('tr', limit=2) [0].findAll('th')]

    rows = soup.findAll('tr')[1:]
    rows_data = [[td.getText() for td in rows[i].findAll('td')] for i in range(len(rows))]
    rows_col_0 = [[th.getText() for th in rows[i].findAll('th')] for i in range(len(rows))]

    rows_col_0.pop(20)
    rows_data.pop(20)

    for i in range(0, len(rows_data)):
        rows_data[i].insert(0,rows_col_0[i])

    headers[7]="OTs"
    headers[3]="Visitor PTS"
    headers[5]="Home PTS"


    frames.append(pd.DataFrame(rows_data, columns=headers, index=range(total_entries, total_entries+len(rows_data))))
    total_entries += len(rows_data)

games = pd.concat(frames)
games = games.drop(["Start (ET)", "Attend.", "Arena", "Notes", '\xa0'], axis=1)

games.drop(games[games["Home PTS"] == ""].index, inplace=True)
visitor_abbr = [None] * len(games)
home_abbr = [None] * len(games)
for i in range(len(games)):
    visitor_abbr[i] = NBA_TEAMS[games["Visitor/Neutral"][i]]
    home_abbr[i] = NBA_TEAMS[games["Home/Neutral"][i]]

games["Home Abbr"] = home_abbr
games["Visitor Abbr"] = visitor_abbr
games["Home PTS"] = pd.to_numeric(games["Home PTS"])
games["Visitor PTS"] = pd.to_numeric(games["Visitor PTS"])




correct_predictions = 0
total_predictions = 0
for i in range(len(games)):
    home = games["Home Abbr"][i]
    away = games["Visitor Abbr"][i]
    home_pts = games["Home PTS"][i]
    away_pts = games["Visitor PTS"][i]
    predicted_home_win = predict_outcome(TEAM_ELO[home], TEAM_ELO[away])
    home_win = True
    if home_pts > away_pts:
        new_elos = calculate_elo(TEAM_ELO[home], TEAM_ELO[away], True)
        TEAM_ELO[home] = new_elos[0]
        TEAM_ELO[away] = new_elos[1]
    else:
        home_win = False
        new_elos = calculate_elo(TEAM_ELO[away], TEAM_ELO[home], False)
        TEAM_ELO[home] = new_elos[1]
        TEAM_ELO[away] = new_elos[0]
    if(i > 442):
        total_predictions += 1
        if(predicted_home_win == home_win):
            correct_predictions += 1

print(f"{correct_predictions} correct out of {total_predictions} : {correct_predictions*100/total_predictions}%")

#Predicting difference in score based on elo
rm_first_n = 100
point_diff = np.zeros((len(games)-rm_first_n, 2))

for i in range(len(games)-rm_first_n):
    home = games["Home Abbr"][i+rm_first_n]
    away = games["Visitor Abbr"][i+rm_first_n]
    home_pts = games["Home PTS"][i+rm_first_n]
    away_pts = games["Visitor PTS"][i+rm_first_n]
    point_diff[i][0] = TEAM_ELO[home]+HOME_ELO_ADVANTAGE - TEAM_ELO[away]
    point_diff[i][1] = home_pts - away_pts

point_diff_df = pd.DataFrame(data=point_diff, columns=["Elo Diff", "Point Diff"])
print(point_diff_df)
point_diff_df.plot.scatter(y="Elo Diff", x="Point Diff")
plt.plot(np.unique(point_diff_df["Point Diff"]), np.poly1d(np.polyfit(point_diff_df["Point Diff"], point_diff_df["Elo Diff"], 1))(np.unique(point_diff_df["Point Diff"])))
plt.show()