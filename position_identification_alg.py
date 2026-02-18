import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal


def create_player_gaussian(player_match_events):
    X = player_match_events[['x', 'y']].values

    mu = X.mean(axis=0) 
    Sigma = np.cov(X, rowvar=False)
    
    return mu, Sigma

def find_box_bounds(player_dists, padding=0.0):
    """
    Compute team bounding box using only Gaussian means (mu).
    Optional padding expands the box by a fixed amount.
    """

    mus = np.vstack(player_dists["mu"].values)  # shape (N, 2)

    team_x_min = mus[:, 0].min() - padding
    team_x_max = mus[:, 0].max() + padding
    team_y_min = mus[:, 1].min() - padding
    team_y_max = mus[:, 1].max() + padding

    return team_x_min, team_y_min, team_x_max, team_y_max

zone_name_grid = [
    ["LB","LWB","LM","LWF","LF"],
    ["LCB","LDM","LCM","LAM","LCF"],
    ["CB","CDM","CM","CAM","CF"],
    ["RCB","RDM","RCM","RAM","RCF"],
    ["RB","RWB","RM","RWF","RF"]
]
def zone_index_to_name(z, nx=5, ny=5):
    ix = z % nx
    iy = z // nx
    return zone_name_grid[ny - 1 -iy][ix]


    
def position_calculations(team_match_events):
    
    # Calculate player distributions
    player_dists = []
    for player in team_match_events.player_id.unique():
        player_match_events = team_match_events[team_match_events.player_id==player]
        mu, Sigma = create_player_gaussian(player_match_events)
        
        player_dist = pd.Series()
        player_dist["player_id"] = player
        player_dist["mu"] = mu
        player_dist["sigma"] = Sigma
        player_dists.append(player_dist)
    player_dists = pd.concat(player_dists,axis=1).T
    
    # Bound the player distributions
    team_x_min, team_y_min, team_x_max, team_y_max = find_box_bounds(player_dists)
    
    # Calculate Probabilities
    nx, ny = 5, 5

    x_edges = np.linspace(team_x_min, team_x_max, nx + 1)
    y_edges = np.linspace(team_y_min, team_y_max, ny + 1)

    def rect_prob(mvn, x0, x1, y0, y1):
        """Probability mass in rectangle using inclusion-exclusion."""
        return (
            mvn.cdf([x1, y1])
            - mvn.cdf([x0, y1])
            - mvn.cdf([x1, y0])
            + mvn.cdf([x0, y0])
        )

    def player_zone_probs(mu, Sigma):
        mvn = multivariate_normal(mean=mu, cov=Sigma)

        zone_probs = np.zeros(nx * ny)

        k = 0
        for iy in range(ny):
            for ix in range(nx):
                x0, x1 = x_edges[ix], x_edges[ix+1]
                y0, y1 = y_edges[iy], y_edges[iy+1]

                zone_probs[k] = rect_prob(mvn, x0, x1, y0, y1)
                k += 1

        # Numerical safety: renormalize
        zone_probs /= zone_probs.sum()

        return zone_probs
    
    zone_rows = []

    for _, row in player_dists.iterrows():
        probs = player_zone_probs(row["mu"], row["sigma"])

        zone_rows.append({
            "player_id": row["player_id"],
            **{f"zone_{i}": probs[i] for i in range(nx*ny)}
        })

    zone_probs_df = pd.DataFrame(zone_rows).set_index("player_id")
    zone_probs_df.columns = list(map(lambda s: zone_index_to_name(int(s.split("_")[1])),zone_probs_df.columns))
    
    # Find highest prob position
    player_pos = zone_probs_df.idxmax(axis=1)
    
    player_dists = player_dists.sort_values("player_id")
    return player_dists.set_index("player_id"), zone_probs_df, player_pos