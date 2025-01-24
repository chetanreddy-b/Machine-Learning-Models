import os

lock_file = "/var/folders/nw/q2mn_x1977g_6z8gkpznh84r0000gp/T/.codecarbon.lock"

# Check if the lock file exists
if os.path.exists(lock_file):
    os.remove(lock_file)
    print(f"Deleted lock file: {lock_file}")
else:
    print("No lock file found.")
