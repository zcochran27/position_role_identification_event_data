# ⚽ Position/Role Identification Algorithm

This repository contains an algorithm for identifying soccer player roles and positional tendencies using event-level tracking data. The method models each player's spatial behavior with Gaussian distributions, discretizes team shape into zones, and assigns the most probable role per player. It also supports comparing **in-possession** vs **out-of-possession** roles and visualizing role transitions.

The approach is designed to work naturally with data from StatsBomb and integrates well with the Python ecosystem.

---

## 📌 Features

- Compute per-player spatial distributions using 2D Gaussians  
- Automatically derive team bounding box and subdivide into zones  
- Map zones to human-readable roles (e.g., LB, CM, RW)  
- Assign most probable role per player  
- Compare in-possession vs out-of-possession roles
- Substitution based algorithm  
- Visualize:
  - Event locations on pitch  
  - Gaussian ellipses  
  - Team bounding boxes and role grids  
  - Player role labels  
  - Role-change arrows between phases  

---

## 🧠 Core Idea

From our event data we want to assign player roles.

![](team_events.png)

1. For each player, fit a 2D Gaussian over their event locations

![](team_gaussians.png)  

2. Compute the team bounding box from player means

![](bound_box.png)

3. Subdivide the box into an `5 × 5` grid  
4. Assign each grid cell a role label  

![](role_assignments.png)

5. Estimate probability of a player occupying each zone  

![](player_probs_map.png)

6. Select the most probable zone as the player’s role  
![](player_roles.png)

--

## 🔁 Phase Based Roles