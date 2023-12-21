import pickle  
import matplotlib.pyplot as plt

import glob
d = r"C:\Users\alyss\Documents\Graduate School\records"
records = glob.glob(r"%s\*"% d)

# records = [
#     r"records\Actual_low_gear"
#            ]

plt.style.use('dark_background')

fig, ax = plt.subplots(1,1)
ax.set_title("Training rewards (average)", weight='semibold')

for record in records:
    file = open(record, 'rb')

    out = pickle.load(file)
    if type(out) == type({}):
        policy_module = out['policy']
        logs = out['logs']
    else:
        policy_module = out

    try:
        ax.plot(logs["reward"])
        # axar[0][0].plot(logs["reward"])
        # axar[0][1].plot(logs["eval step_count"])
        # axar[1][0].plot(logs["eval reward (sum)"])
        # axar[1][1].plot(logs["step_count"])
    except:
        print(record)
    
# ax.legend()
plt.show()