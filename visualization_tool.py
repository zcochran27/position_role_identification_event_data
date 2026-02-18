import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch, Circle
from mplsoccer import Pitch
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
from position_identification_alg import find_box_bounds, zone_index_to_name, zone_name_grid

def plot_team_events(team_match_events, ax = None):
    pitch = Pitch(pitch_type="custom",
                      pitch_length=120,
                      pitch_width=80,
                      line_color="black",
                      pitch_color="white")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7))
    else:
        fig = ax.figure

    pitch.draw(ax=ax)
    for p, group in team_match_events.groupby("player_id"):
        ax.scatter(group["x"],group["y"],label=p)
    ax.legend(title = "Players")
    ax.set_title("Team Events")
    return fig, ax
    
def plot_player_events(player_match_events,ax=None):
    pitch = Pitch(pitch_type="custom",
                      pitch_length=120,
                      pitch_width=80,
                      line_color="black",
                      pitch_color="white")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7))
    else:
        fig = ax.figure

    pitch.draw(ax=ax)
    sns.scatterplot(data=player_match_events,x="x",y="y",ax=ax)
    return fig, ax
    
def create_gaussian_ellipse(mu, Sigma, se = .5):
    eigvals, eigvecs = np.linalg.eigh(Sigma)

    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    width, height = 2 * se * np.sqrt(eigvals)  

    angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
    
    ellipse = Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=angle,
        edgecolor='blue',
        facecolor='none',
        linewidth=1,
        alpha = .25
    )
    
    return ellipse
    
    
def plot_player_gaussian(player_match_events, mu, Sigma, plot_events=True, ax=None):
    pitch = Pitch(pitch_type="custom",
                      pitch_length=120,
                      pitch_width=80,
                      line_color="black",
                      pitch_color="white")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7))
    else:
        fig = ax.figure

    pitch.draw(ax=ax)
    if plot_events:
        sns.scatterplot(data=player_match_events,x="x",y="y",ax=ax)

    ellipse = create_gaussian_ellipse(mu, Sigma)
    ax.add_patch(ellipse)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax

def plot_team_player_gaussians(player_dists,with_box=False, with_cells=False,nx=5,ny=5, ax=None):
    pitch = Pitch(pitch_type="custom",
                      pitch_length=120,
                      pitch_width=80,
                      line_color="black",
                      pitch_color="white")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7))
    else:
        fig = ax.figure

    pitch.draw(ax=ax)

    for _, row in player_dists.iterrows():
        mu = row["mu"]
        Sigma = row["sigma"]

        ellipse = create_gaussian_ellipse(mu,Sigma)

        player_id = row.name

        ax.add_patch(ellipse)
        ax.scatter(mu[0], mu[1], s=25, label=str(int(player_id)))
    
    ax.legend(title="player_id", loc="upper left", fontsize=8)
    
    if with_box:
        team_x_min, team_y_min, team_x_max, team_y_max = find_box_bounds(player_dists)
        rect = Rectangle(
            (team_x_min, team_y_min),
            team_x_max - team_x_min,
            team_y_max - team_y_min,
            fill=False,
            edgecolor="red",
            linewidth=2
        )
        ax.add_patch(rect)
        if with_cells:
            dx = (team_x_max - team_x_min) / nx
            dy = (team_y_max - team_y_min) / ny

            # vertical grid lines
            for i in range(1, nx):
                x = team_x_min + i * dx
                ax.plot([x, x], [team_y_min, team_y_max], color="black", linewidth=1)

            # horizontal grid lines
            for j in range(1, ny):
                y = team_y_min + j * dy
                ax.plot([team_x_min, team_x_max], [y, y], color="black", linewidth=1)
            
            # ---- ADD ROLE LABELS ----
            for z in range(nx * ny):
                ix = z % nx
                iy = z // nx

                role = zone_index_to_name(z)

                cx = team_x_min + (ix + 0.5) * dx
                cy = team_y_min + (iy + 0.5) * dy

                ax.text(cx, cy, role,
                        ha="center", va="center",
                        fontsize=9,
                        color="gray",
                        alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Player Event Dists")

    return fig, ax
    
def plot_role_assignments(nx=5,ny=5, ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,6))
    else:
        fig = ax.figure

    for i in range(nx + 1):
        ax.plot([i, i], [0, ny], color="black")
    for j in range(ny + 1):
        ax.plot([0, nx], [j, j], color="black")

    for z in range(nx*ny):
        ix = z % nx    
        iy = z // nx      
        
        zone_name = zone_index_to_name(z)

        ax.text(ix + 0.5, iy + 0.5,
                str(zone_name),
                ha="center", va="center", fontsize=10)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Player Position/Role Assignment")

    return fig, ax

def plot_player_gaussian_with_zones(player_id, mu, Sigma, zone_probs,
                                    team_x_min, team_x_max,
                                    team_y_min, team_y_max,
                                    nx=5, ny=5, ax = None):
    pitch = Pitch(pitch_type="custom",
                      pitch_length=120,
                      pitch_width=80,
                      line_color="black",
                      pitch_color="white")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7))
    else:
        fig = ax.figure

    pitch.draw(ax=ax)

    rect = Rectangle(
        (team_x_min, team_y_min),
        team_x_max - team_x_min,
        team_y_max - team_y_min,
        fill=False, edgecolor="red", linewidth=2
    )
    ax.add_patch(rect)

    dx = (team_x_max - team_x_min) / nx
    dy = (team_y_max - team_y_min) / ny

    for i in range(1, nx):
        x = team_x_min + i*dx
        ax.plot([x,x],[team_y_min,team_y_max],color="black",lw=1)
    for j in range(1, ny):
        y = team_y_min + j*dy
        ax.plot([team_x_min,team_x_max],[y,y],color="black",lw=1)

    ell = create_gaussian_ellipse(mu,Sigma)
    ax.add_patch(ell)

    ax.scatter(mu[0], mu[1], color="blue", s=40, zorder=3)

    for iy in range(ny):
        for ix in range(nx):
            zone_id = iy*nx + ix
            p = zone_probs[zone_id]

            cx = team_x_min + (ix+0.5)*dx
            cy = team_y_min + (iy+0.5)*dy

            ax.text(cx, cy, f"{p:.2f}",
                    ha="center", va="center", fontsize=9)

    ax.set_title(f"Player {player_id} Gaussian Distribution with Zone Probabilities")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return fig, ax
    
def plot_player_probable_position(player_pos, team_subs, ny=5, nx=5,use_subbing = True, ax = None):

    pos_to_coord = {}
    for iy in range(ny):
        for ix in range(nx):
            pos_to_coord[zone_name_grid[ny-1-iy][ix]] = (ix, iy)

    came_on = set(team_subs["substitution_replacement_id"].dropna().astype(int))
    came_off = set(team_subs["player_id"].dropna().astype(int))

    both = came_on & came_off

    cell_players = defaultdict(list)
    for player_id, pos in player_pos.items():
        cell_players[pos].append(int(player_id))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7,6))
    else:
        fig = ax.figure

    for i in range(nx + 1):
        ax.plot([i, i], [0, ny], color="black")
    for j in range(ny + 1):
        ax.plot([0, nx], [j, j], color="black")

    for pos, players in cell_players.items():
        ix, iy = pos_to_coord[pos]
        n = len(players)
        offsets = np.linspace(0.7, 0.3, n)

        for pid, off in zip(players, offsets):

            if use_subbing:
                if pid in both:
                    color = "orange"
                elif pid in came_on:
                    color = "green"
                elif pid in came_off:
                    color = "red"
                else:
                    color = "black"
            else:
                color = "black"

            ax.text(ix + 0.5, iy + off,
                    str(pid),
                    ha="center", va="center",
                    fontsize=9,
                    color=color)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Players by Most Probable Role")
    ax.text(0, -.2, "Green: Sub On   Red: Sub Off   Orange: Sub On+Off", fontsize=9)
    
    return fig, ax

def plot_player_role_probabilities(player_dists, zone_probs_df, player_id, nx=5,ny=5, ax = None):
    player_dist = player_dists.loc[player_id]
    pitch = Pitch(pitch_type="custom",
                      pitch_length=120,
                      pitch_width=80,
                      line_color="black",
                      pitch_color="white")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7))
    else:
        fig = ax.figure

    pitch.draw(ax=ax)

    mu = player_dist["mu"]
    Sigma = player_dist["sigma"]
    ellipse = create_gaussian_ellipse(mu,Sigma)
    ax.add_patch(ellipse)
    ax.scatter(mu[0], mu[1], s=25)

    team_x_min, team_y_min, team_x_max, team_y_max = find_box_bounds(player_dists)
    rect = Rectangle(
        (team_x_min, team_y_min),
        team_x_max - team_x_min,
        team_y_max - team_y_min,
        fill=False,
        edgecolor="red",
        linewidth=2
    )
    ax.add_patch(rect)

    dx = (team_x_max - team_x_min) / nx
    dy = (team_y_max - team_y_min) / ny

    for i in range(1, nx):
        x = team_x_min + i * dx
        ax.plot([x, x], [team_y_min, team_y_max], color="black", linewidth=1)

    for j in range(1, ny):
        y = team_y_min + j * dy
        ax.plot([team_x_min, team_x_max], [y, y], color="black", linewidth=1)

    for z in range(nx * ny):
        ix = z % nx
        iy = z // nx

        prob = round(zone_probs_df.loc[player_id].iloc[z],2)

        cx = team_x_min + (ix + 0.5) * dx
        cy = team_y_min + (iy + 0.5) * dy

        ax.text(cx, cy, prob,
                ha="center", va="center",
                fontsize=9,
                color="black",
                alpha=0.8)
        
        ax.set_title(f"Player {player_id} Role Probabilities")
    
    return fig, ax

from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.cm import get_cmap
from position_identification_alg import zone_name_grid

def plot_role_transition_arrows(
    player_pos_in,
    player_pos_out,
    zone_name_grid,
    ny=5,
    nx=5,
    ax=None
):
    """
    Draw arrows from in-possession role to out-of-possession role.
    Players are stacked vertically inside each cell.
    Circle indicates no change in role.
    """

    pos_to_coord = {}
    for iy in range(ny):
        for ix in range(nx):
            pos_to_coord[zone_name_grid[ny-1-iy][ix]] = (ix, iy)

    player_ids = sorted(
        set(player_pos_in.keys()) | set(player_pos_out.keys())
    )

    cmap = plt.cm.tab20
    color_map = {
        pid: cmap(i % 20) for i, pid in enumerate(player_ids)
    }

    cell_players = defaultdict(list)
    for pid in player_ids:
        role = player_pos_in.get(pid)
        if role in pos_to_coord:
            cell_players[role].append(pid)


    if ax is None:
        fig, ax = plt.subplots(figsize=(7,6))
    else:
        fig = ax.figure

    for i in range(nx + 1):
        ax.plot([i, i], [0, ny], color="black")
    for j in range(ny + 1):
        ax.plot([0, nx], [j, j], color="black")


    for role, pids in cell_players.items():

        ix, iy = pos_to_coord[role]
        cx, cy = ix + 0.5, iy + 0.5

        spacing = 0.15
        start_y = cy + (len(pids)-1)/2 * spacing

        for i, pid in enumerate(pids):

            in_role = player_pos_in.get(pid)
            out_role = player_pos_out.get(pid)

            if out_role not in pos_to_coord:
                continue

            x2, y2 = pos_to_coord[out_role]
            end = (x2 + 0.5, y2 + 0.5)

            y_offset = start_y - i * spacing
            start = (cx, y_offset)

            color = color_map[pid]

            # No role change → circle
            if in_role == out_role:
                circ = Circle(start, 0.12,
                              edgecolor=color,
                              facecolor="none",
                              lw=2)
                ax.add_patch(circ)

            # Role change → arrow
            else:
                arrow = FancyArrowPatch(
                    start, end,
                    arrowstyle="->",
                    lw=2,
                    color=color,
                    mutation_scale=12
                )
                ax.add_patch(arrow)

            ax.text(cx, y_offset,
                    str(pid),
                    ha="center", va="center",
                    fontsize=8,
                    color=color)


    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("In → Out of Possession Role Transitions")

    return fig, ax