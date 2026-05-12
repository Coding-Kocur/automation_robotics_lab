
# =============================================================================
#  AUTOMATION & ROBOTICS -- COMPLETE LAB SOLUTIONS (Labs 1-6)
#  Open in VS Code and run with Ctrl+F5, or copy cells into Jupyter Notebook.
#
#  Install dependencies first:
#    pip install pulp numpy pandas matplotlib scipy scikit-learn networkx criticalpath sktime prophet
# =============================================================================
import matplotlib
matplotlib.use("Agg")   # Save PNGs without needing a display window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ===========================================================================
#  LAB 1 -- Python Basics & Introduction to PuLP (Linear Programming)
# ===========================================================================

# ---- 1.1  Basic Python recap -----------------------------------------------
x = 5
y = 3.14
name = "Automation"
print(f"x={x}, y={y}, name={name}")

numbers = [1, 2, 3, 4, 5]
person  = {"name": "Alice", "age": 30}
coords  = (10.0, 20.0)

squares = [n**2 for n in range(1, 6)]
print("Squares:", squares)

# ---- 1.2  Simple LP -- maximise profit -------------------------------------
import pulp

prob = pulp.LpProblem("Simple_LP", pulp.LpMaximize)
x1 = pulp.LpVariable("x1", lowBound=0)
x2 = pulp.LpVariable("x2", lowBound=0)

prob += 3*x1 + 5*x2, "Profit"
prob += x1 <= 4,            "C1_x1_limit"
prob += 2*x2 <= 12,         "C2_x2_limit"
prob += 3*x1 + 5*x2 <= 25, "C3_resource"

prob.solve(pulp.PULP_CBC_CMD(msg=0))

print("\n=== Lab 1 -- Simple LP ===")
print("Status:", pulp.LpStatus[prob.status])
print(f"x1={pulp.value(x1):.2f}, x2={pulp.value(x2):.2f}")
print(f"Max Profit = {pulp.value(prob.objective):.2f}")

# ---- 1.3  File I/O example -------------------------------------------------
import csv, os

data = [["Product", "Qty", "Price"],
        ["Chair",   10,    150],
        ["Table",    5,    400]]

csv_path = "lab1_output.csv"
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerows(data)

with open(csv_path, "r") as f:
    print("CSV contents:")
    print(f.read())

os.remove(csv_path)


# ===========================================================================
#  LAB 2 -- Advanced Linear Programming (Production Optimisation)
# ===========================================================================

# ---- 2.1  Production problem -- Beds & Chairs ------------------------------
prob2  = pulp.LpProblem("Production_Beds_Chairs", pulp.LpMaximize)
beds   = pulp.LpVariable("beds",   lowBound=0, cat="Integer")
chairs = pulp.LpVariable("chairs", lowBound=0, cat="Integer")

prob2 += 600*beds + 400*chairs, "Total_Profit"
prob2 += 3*beds +  2*chairs <= 120, "Wood_m2"
prob2 += 4*beds +  3*chairs <= 160, "Labour_h"
prob2 += 1*beds + 0.5*chairs <= 35, "Paint_L"

prob2.solve(pulp.PULP_CBC_CMD(msg=0))

print("\n=== Lab 2 -- Production Optimisation ===")
print("Status:", pulp.LpStatus[prob2.status])
print(f"Beds   = {pulp.value(beds):.0f}")
print(f"Chairs = {pulp.value(chairs):.0f}")
print(f"Max Profit = {pulp.value(prob2.objective):.2f} PLN")

print("\nConstraint slack values:")
for cname, constraint in prob2.constraints.items():
    print(f"  {cname}: slack = {constraint.slack:.2f}")

# ---- 2.2  Diet problem -- minimise cost ------------------------------------
prob_diet = pulp.LpProblem("Diet_Problem", pulp.LpMinimize)
bread = pulp.LpVariable("bread", lowBound=0)
milk  = pulp.LpVariable("milk",  lowBound=0)
eggs  = pulp.LpVariable("eggs",  lowBound=0)

prob_diet += 2.5*bread + 3.5*milk + 5.0*eggs, "Cost"
prob_diet += 3*bread + 8*milk + 6*eggs >= 55,  "Protein"
prob_diet += 6*bread + 2*milk + 4*eggs >= 33,  "Carbs"
prob_diet += 2*bread + 3*milk + 5*eggs >= 30,  "Fat"

prob_diet.solve(pulp.PULP_CBC_CMD(msg=0))

print("\n=== Lab 2 -- Diet Problem ===")
print(f"Bread={pulp.value(bread):.2f}, Milk={pulp.value(milk):.2f}, Eggs={pulp.value(eggs):.2f}")
print(f"Min Cost = {pulp.value(prob_diet.objective):.2f}")


# ===========================================================================
#  LAB 3 -- Integer Programming & Binary / Facility Location
# ===========================================================================

# ---- 3.1  0-1 Knapsack -----------------------------------------------------
items    = ["laptop","camera","book","headphones","jacket"]
weights  = [3, 1, 0.5, 0.3, 1.2]
values   = [1500, 800, 200, 150, 300]
capacity = 4.0

prob_ks = pulp.LpProblem("Knapsack", pulp.LpMaximize)
xk = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(items))]

prob_ks += pulp.lpSum(values[i]*xk[i]  for i in range(len(items))), "Value"
prob_ks += pulp.lpSum(weights[i]*xk[i] for i in range(len(items))) <= capacity, "Capacity"
prob_ks.solve(pulp.PULP_CBC_CMD(msg=0))

print("\n=== Lab 3 -- 0-1 Knapsack ===")
for i, item in enumerate(items):
    if pulp.value(xk[i]) > 0.5:
        print(f"  Pack: {item}  (value={values[i]}, weight={weights[i]})")
print(f"Total Value = {pulp.value(prob_ks.objective):.0f}")

# ---- 3.2  Facility Location (Big-M) ----------------------------------------
import numpy as np

num_customers  = 4
num_facilities = 3
open_cost = [500, 400, 600]
transport_cost = np.array([
    [10, 20, 30, 40],
    [15, 10, 25, 35],
    [20, 15, 10, 20],
])

prob_fl = pulp.LpProblem("Facility_Location", pulp.LpMinimize)
yf = [pulp.LpVariable(f"open_{j}", cat="Binary") for j in range(num_facilities)]
zf = [[pulp.LpVariable(f"assign_{j}_{i}", cat="Binary")
       for i in range(num_customers)] for j in range(num_facilities)]

prob_fl += (pulp.lpSum(open_cost[j]*yf[j] for j in range(num_facilities)) +
            pulp.lpSum(transport_cost[j][i]*zf[j][i]
                       for j in range(num_facilities)
                       for i in range(num_customers)), "Total_Cost")

for i in range(num_customers):
    prob_fl += pulp.lpSum(zf[j][i] for j in range(num_facilities)) == 1

for j in range(num_facilities):
    for i in range(num_customers):
        prob_fl += zf[j][i] <= yf[j]

prob_fl.solve(pulp.PULP_CBC_CMD(msg=0))

print("\n=== Lab 3 -- Facility Location ===")
for j in range(num_facilities):
    if pulp.value(yf[j]) > 0.5:
        assigned = [i for i in range(num_customers) if pulp.value(zf[j][i]) > 0.5]
        print(f"  Open Facility {j}, serves customers {assigned}")
print(f"Min Total Cost = {pulp.value(prob_fl.objective):.0f}")

# ---- 3.3  Single-machine scheduling (Big-M precedence) ---------------------
jobs_sm   = [0, 1, 2]
durations = [3, 5, 2]
M_large   = 1000

prob_sched = pulp.LpProblem("Single_Machine_Schedule", pulp.LpMinimize)
start_sm   = [pulp.LpVariable(f"start_{j}", lowBound=0) for j in jobs_sm]
precede    = [[pulp.LpVariable(f"prec_{j}_{k}", cat="Binary")
               for k in jobs_sm] for j in jobs_sm]
Cmax_sm    = pulp.LpVariable("C_max", lowBound=0)

prob_sched += Cmax_sm, "Makespan"
for j in jobs_sm:
    prob_sched += start_sm[j] + durations[j] <= Cmax_sm
for j in jobs_sm:
    for k in jobs_sm:
        if j != k:
            prob_sched += start_sm[j] + durations[j] <= start_sm[k] + M_large*(1 - precede[j][k])
            prob_sched += precede[j][k] + precede[k][j] == 1

prob_sched.solve(pulp.PULP_CBC_CMD(msg=0))

print("\n=== Lab 3 -- Single-Machine Scheduling ===")
for j in jobs_sm:
    print(f"  Job {j}: start={pulp.value(start_sm[j]):.1f}, "
          f"end={pulp.value(start_sm[j])+durations[j]:.1f}")
print(f"Makespan = {pulp.value(Cmax_sm):.1f}")


# ===========================================================================
#  LAB 4 -- Job-Shop Scheduling (Minimise Makespan)
# ===========================================================================

jobs_data = {
    0: [(0, 3), (1, 2), (2, 2)],
    1: [(0, 2), (2, 1), (1, 4)],
    2: [(1, 4), (0, 3)],
}
M_big = 1000

prob_js = pulp.LpProblem("JobShop", pulp.LpMinimize)

start_js = {(j, k): pulp.LpVariable(f"s_{j}_{k}", lowBound=0)
            for j in jobs_data for k in range(len(jobs_data[j]))}

order = {}
for j1 in jobs_data:
    for j2 in jobs_data:
        if j1 < j2:
            for k1, (m1, _) in enumerate(jobs_data[j1]):
                for k2, (m2, _) in enumerate(jobs_data[j2]):
                    if m1 == m2:
                        order[(j1,k1,j2,k2)] = pulp.LpVariable(
                            f"ord_{j1}_{k1}_{j2}_{k2}", cat="Binary")

Cmax_js = pulp.LpVariable("C_max", lowBound=0)
prob_js += Cmax_js

for j, ops in jobs_data.items():
    for k in range(len(ops)-1):
        _, dur = ops[k]
        prob_js += start_js[(j,k)] + dur <= start_js[(j,k+1)]

for j, ops in jobs_data.items():
    for k, (m, dur) in enumerate(ops):
        prob_js += start_js[(j,k)] + dur <= Cmax_js

for (j1,k1,j2,k2), var in order.items():
    _, dur1 = jobs_data[j1][k1]
    _, dur2 = jobs_data[j2][k2]
    prob_js += start_js[(j1,k1)] + dur1 <= start_js[(j2,k2)] + M_big*(1-var)
    prob_js += start_js[(j2,k2)] + dur2 <= start_js[(j1,k1)] + M_big*var

prob_js.solve(pulp.PULP_CBC_CMD(msg=0))

print("\n=== Lab 4 -- Job-Shop Scheduling ===")
print(f"Makespan = {pulp.value(Cmax_js):.1f}")
for j, ops in jobs_data.items():
    for k, (m, dur) in enumerate(ops):
        s = pulp.value(start_js[(j,k)])
        print(f"  Job {j}, Op {k}: Machine {m}, start={s:.1f}, end={s+dur:.1f}")

# Gantt chart
colors = ["#4C72B0","#DD8452","#55A868"]
fig, ax = plt.subplots(figsize=(10,4))
for j, ops in jobs_data.items():
    for k, (m, dur) in enumerate(ops):
        s = pulp.value(start_js[(j,k)])
        ax.broken_barh([(s, dur)], (m*2, 1.5),
                       facecolors=colors[j], edgecolor="black", linewidth=0.5)
        ax.text(s + dur/2, m*2+0.75, f"J{j}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
ax.set_yticks([0.75, 2.75, 4.75])
ax.set_yticklabels(["Machine 0","Machine 1","Machine 2"])
ax.set_xlabel("Time")
ax.set_title("Lab 4 -- Job-Shop Gantt Chart")
patches = [mpatches.Patch(color=colors[j], label=f"Job {j}") for j in jobs_data]
ax.legend(handles=patches, loc="upper right")
plt.tight_layout()
plt.savefig("lab4_gantt.png", dpi=100)
plt.close()
print("Gantt chart saved -> lab4_gantt.png")


# ===========================================================================
#  LAB 5 -- Critical Path Method (CPM) & PERT
# ===========================================================================

import networkx as nx

activities = [
    ("A", 3, []),
    ("B", 4, []),
    ("C", 2, ["A"]),
    ("D", 5, ["B"]),
    ("E", 3, ["C","D"]),
    ("F", 2, ["E"]),
    ("G", 4, ["E"]),
    ("H", 3, ["F","G"]),
]

G = nx.DiGraph()
for aname, dur, preds in activities:
    G.add_node(aname, duration=dur)
    for pred in preds:
        G.add_edge(pred, aname)

# Forward pass
ES = {}; EF = {}
for node in nx.topological_sort(G):
    p = list(G.predecessors(node))
    ES[node] = max((EF[q] for q in p), default=0)
    EF[node] = ES[node] + G.nodes[node]["duration"]

proj_dur = max(EF.values())

# Backward pass
LS = {}; LF = {}
for node in reversed(list(nx.topological_sort(G))):
    s = list(G.successors(node))
    LF[node] = min((LS[q] for q in s), default=proj_dur)
    LS[node] = LF[node] - G.nodes[node]["duration"]

print("\n=== Lab 5 -- CPM ===")
print(f"Project Duration = {proj_dur}")
print(f"{'Task':<6} {'ES':>4} {'EF':>4} {'LS':>4} {'LF':>4} {'Float':>6} {'Critical':>9}")
critical_path = []
for node in nx.topological_sort(G):
    tf = LS[node] - ES[node]
    if tf == 0:
        critical_path.append(node)
    print(f"{node:<6} {ES[node]:>4} {EF[node]:>4} {LS[node]:>4} {LF[node]:>4} "
          f"{tf:>6} {'*' if tf==0 else '':>9}")
print(f"Critical Path: {' -> '.join(critical_path)}")

# Draw CPM network
plt.figure(figsize=(10,6))
pos = nx.spring_layout(G, seed=42)
node_colors = ["red" if n in critical_path else "lightblue" for n in G.nodes()]
nx.draw(G, pos, with_labels=True, node_color=node_colors,
        node_size=1500, font_size=12, arrows=True,
        edge_color="gray", font_color="black")
plt.title("Lab 5 -- CPM Network (red = critical)")
plt.tight_layout()
plt.savefig("lab5_cpm.png", dpi=100)
plt.close()
print("CPM network saved -> lab5_cpm.png")

# PERT three-point estimates
pert_activities = {
    "A": (1, 3, 5), "B": (2, 4, 6), "C": (1, 2, 3),
    "D": (3, 5, 9), "E": (2, 3, 4), "F": (1, 2, 3),
}

print("\n=== Lab 5 -- PERT Estimates ===")
print(f"{'Task':<6} {'a':>4} {'m':>4} {'b':>4} {'Expected':>10} {'Variance':>10} {'StdDev':>10}")
for task, (a, m, b) in pert_activities.items():
    te  = (a + 4*m + b) / 6.0
    var = ((b - a) / 6.0) ** 2
    print(f"{task:<6} {a:>4} {m:>4} {b:>4} {te:>10.2f} {var:>10.4f} {var**0.5:>10.4f}")

# criticalpath library (optional)
try:
    from criticalpath import Node
    project = Node("Project")
    tnodes  = {}
    for aname, dur, preds in activities:
        t = Node(aname, duration=dur)
        tnodes[aname] = t
        project.add(t)
    for aname, dur, preds in activities:
        for pred in preds:
            tnodes[aname].add(tnodes[pred])
    project.update_all()
    print("\n=== Lab 5 -- criticalpath library ===")
    print("Critical path:", [n.name for n in project.get_critical_path()])
    print("Duration:", project.duration)
except ImportError:
    print("\ncriticalpath library not installed; skipping.")


# ===========================================================================
#  LAB 6 -- Machine Learning & Statistics in Automation
# ===========================================================================

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---- 6.1  Time-Series Forecasting with sktime (NaiveForecaster) ------------
try:
    from sktime.datasets import load_airline
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.utils.plotting import plot_series

    y_ts = load_airline()
    fh   = np.arange(1, 37)

    forecaster = NaiveForecaster(strategy="last", sp=12)
    forecaster.fit(y_ts)
    y_pred_ts = forecaster.predict(fh)

    fig, ax = plot_series(y_ts, y_pred_ts, labels=["Actual", "Forecast"])
    ax.set_title("Lab 6 -- Airline Passengers: NaiveForecaster")
    plt.tight_layout()
    plt.savefig("lab6_naive_forecast.png", dpi=100)
    plt.close()
    print("\n=== Lab 6 -- NaiveForecaster (sktime) ===")
    print("Last 5 predictions:", y_pred_ts.values[-5:])
    print("Saved -> lab6_naive_forecast.png")
except (ImportError, ModuleNotFoundError):
    print("\nsktime (or seaborn) not installed; skipping NaiveForecaster demo.")

# ---- 6.2  Time-Series Forecasting with Prophet -----------------------------
try:
    from prophet import Prophet

    dates = pd.date_range("2020-01-01", periods=104, freq="W")
    sales = 200 + 50*np.sin(np.linspace(0, 4*np.pi, 104)) + \
            np.random.normal(0, 10, 104)
    df_p = pd.DataFrame({"ds": dates, "y": sales})

    mp = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                 daily_seasonality=False)
    mp.fit(df_p)

    future   = mp.make_future_dataframe(periods=26, freq="W")
    forecast = mp.predict(future)

    fig = mp.plot(forecast)
    fig.suptitle("Lab 6 -- Prophet Sales Forecast")
    plt.savefig("lab6_prophet_forecast.png", dpi=100)
    plt.close()

    print("\n=== Lab 6 -- Prophet Forecast ===")
    print(forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(5).to_string(index=False))
    print("Saved -> lab6_prophet_forecast.png")
except ImportError:
    print("\nProphet not installed; skipping Prophet demo.")

# ---- 6.3  Machine Condition Classification ---------------------------------
np.random.seed(42)
n_samples = 200
vibration   = np.concatenate([np.random.normal(0.5, 0.1, n_samples//2),
                               np.random.normal(1.5, 0.3, n_samples//2)])
temperature = np.concatenate([np.random.normal(60, 5, n_samples//2),
                               np.random.normal(80, 10, n_samples//2)])
y_class = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
X = np.column_stack([vibration, temperature])

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.25, random_state=42)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)
y_pred_cls = clf.predict(X_test)

print("\n=== Lab 6 -- Machine Condition Classification ===")
print(classification_report(y_test, y_pred_cls, target_names=["Healthy","Faulty"]))

plt.figure(figsize=(8,5))
for lab, col, lbl in [(0,"#4C72B0","Healthy"),(1,"#DD8452","Faulty")]:
    mask = y_class == lab
    plt.scatter(X[mask,0], X[mask,1], c=col, label=lbl, alpha=0.6, edgecolors="k")
plt.xlabel("Vibration"); plt.ylabel("Temperature (C)")
plt.title("Lab 6 -- Machine Health: Vibration vs Temperature")
plt.legend(); plt.tight_layout()
plt.savefig("lab6_classification.png", dpi=100)
plt.close()
print("Saved -> lab6_classification.png")

# ---- 6.4  K-Means Clustering -----------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

plt.figure(figsize=(8,5))
for c in [0,1]:
    mask = cluster_labels == c
    plt.scatter(X[mask,0], X[mask,1], label=f"Cluster {c}", alpha=0.6, edgecolors="k")
plt.xlabel("Vibration"); plt.ylabel("Temperature (C)")
plt.title("Lab 6 -- K-Means Clustering of Machine Data")
plt.legend(); plt.tight_layout()
plt.savefig("lab6_kmeans.png", dpi=100)
plt.close()

print("\n=== Lab 6 -- K-Means Clustering ===")
print("Cluster sizes:", {c: int((cluster_labels==c).sum()) for c in [0,1]})
print("Saved -> lab6_kmeans.png")

# ---- 6.5  Linear Regression on sensor data ---------------------------------
vib_level  = np.random.uniform(0.2, 2.5, 80).reshape(-1,1)
maint_cost = 100 + 150*vib_level.ravel() + np.random.normal(0, 20, 80)

reg = LinearRegression()
reg.fit(vib_level, maint_cost)
y_pred_reg = reg.predict(vib_level)

print("\n=== Lab 6 -- Linear Regression (Vibration -> Maintenance Cost) ===")
print(f"Coefficient : {reg.coef_[0]:.2f}")
print(f"Intercept   : {reg.intercept_:.2f}")
print(f"MAE  = {mean_absolute_error(maint_cost, y_pred_reg):.2f}")
rmse = mean_squared_error(maint_cost, y_pred_reg) ** 0.5
print(f"RMSE = {rmse:.2f}")

plt.figure(figsize=(8,5))
plt.scatter(vib_level, maint_cost, alpha=0.5, label="Observations")
plt.plot(np.sort(vib_level.ravel()), reg.predict(np.sort(vib_level)),
         color="red", linewidth=2, label="Regression line")
plt.xlabel("Vibration Level"); plt.ylabel("Maintenance Cost (PLN)")
plt.title("Lab 6 -- Vibration vs Maintenance Cost")
plt.legend(); plt.tight_layout()
plt.savefig("lab6_regression.png", dpi=100)
plt.close()
print("Saved -> lab6_regression.png")

# ===========================================================================
print("\n=== All Labs Complete! ===")
print("Generated PNG files in current directory:")
for f in ["lab4_gantt.png","lab5_cpm.png",
          "lab6_naive_forecast.png","lab6_prophet_forecast.png",
          "lab6_classification.png","lab6_kmeans.png","lab6_regression.png"]:
    print(f"  {f}")
