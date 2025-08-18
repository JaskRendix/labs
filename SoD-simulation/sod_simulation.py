from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# -------------------------------
# Generate synthetic users and roles
# -------------------------------
np.random.seed(42)
users = [f"user_{i}" for i in range(1, 101)]
roles = [
    "Payment Approval",
    "Vendor Management",
    "Inventory Control",
    "Procurement",
    "Financial Reporting",
    "System Administration",
]

user_roles = []
for user in users:
    assigned = np.random.choice(roles, size=np.random.randint(1, 4), replace=False)
    for role in assigned:
        user_roles.append({"user": user, "role": role})

df_roles = pd.DataFrame(user_roles)
df_roles.to_csv("user_roles.csv", index=False)

# -------------------------------
# Define conflicting role pairs (SoD risks)
# -------------------------------
conflicts = [
    ("Payment Approval", "Vendor Management"),
    ("System Administration", "Financial Reporting"),
    ("Procurement", "Inventory Control"),
]

# -------------------------------
# Flag users with conflicting roles
# -------------------------------
flags = []
for user in df_roles["user"].unique():
    user_set = set(df_roles[df_roles["user"] == user]["role"])
    for r1, r2 in conflicts:
        if r1 in user_set and r2 in user_set:
            flags.append({"user": user, "conflict": f"{r1} & {r2}"})

df_conflicts = pd.DataFrame(flags)
df_conflicts.to_csv("segregation_of_duties_risks.csv", index=False)

# -------------------------------
# Simulate login events
# -------------------------------
login_data = []

for user in users:
    num_logins = np.random.randint(1, 6)
    for _ in range(num_logins):
        days_ago = np.random.randint(0, 30)
        time_offset = timedelta(
            days=days_ago,
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
        )
        timestamp = datetime.now() - time_offset

        ip = f"192.168.{np.random.randint(0,255)}.{np.random.randint(0,255)}"
        location = np.random.choice(["Zurich", "London", "New York", "Tokyo", "Sydney"])

        login_data.append(
            {
                "user": user,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "ip_address": ip,
                "location": location,
            }
        )

df_logins = pd.DataFrame(login_data)
df_logins.to_csv("user_login_events.csv", index=False)
