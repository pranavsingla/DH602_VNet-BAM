import re
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize lists to store iteration and cons_dist values
iterations = []
cons_dist_values = []

# Regex pattern to match iteration and cons_dist values in the log file
pattern = r'iteration (\d+) : loss : [\d.]+ cons_dist: ([\d.]+)'

# Path to the log file
log_file_path = '/home/healthcrafters/healthcrafters/UA-MT/code/uamtbamlabel.out'

# Read the log file line by line
with open(log_file_path, 'r') as file:
    for line in file:
        # Search for the pattern in each line
        match = re.search(pattern, line)
        if match:
            # Extract iteration number and cons_dist value
            iteration = int(match.group(1))
            cons_dist = float(match.group(2))

            # Append to the lists
            iterations.append(iteration)
            cons_dist_values.append(cons_dist)

# Plot using Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(x=iterations, y=cons_dist_values)
plt.xlabel('Iterations')
plt.ylabel('Cons Dist')
plt.title('Cons Dist vs Iterations')
plt.show()


# import re
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Initialize lists to store iteration and loss values
# iterations = []
# loss_values = []

# # Regex pattern to match iteration and loss values in the log file
# pattern = r'iteration (\d+) : loss : ([\d.]+)'

# # Path to the log file
# log_file_path = '/home/healthcrafters/healthcrafters/UA-MT/code/uamtbamlabel.out'

# # Read the log file line by line
# with open(log_file_path, 'r') as file:
#     for line in file:
#         # Search for the pattern in each line
#         match = re.search(pattern, line)
#         if match:
#             # Extract iteration number and loss value
#             iteration = int(match.group(1))
#             loss = float(match.group(2))

#             # Append to the lists
#             iterations.append(iteration)
#             loss_values.append(loss)

# # Plot using Seaborn
# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))
# sns.lineplot(x=iterations, y=loss_values)
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Validation Loss vs Iterations')
# plt.show()

# # import matplotlib.pyplot as plt

# # def parse_log_file(log_file):
# #     curves = {}
# #     current_curve = []
# #     with open(log_file, 'r') as f:
# #         for line in f:
# #             if line.startswith('iteration'):
# #                 parts = line.split(':')
# #                 print(parts)
# #                 iteration = int(parts[0].split()[1])
# #                 loss = float(parts[1].strip().split()[0])
# #                 current_curve.append((iteration, loss))
# #             elif line.startswith("total") and "samples" in line:
# #                 if current_curve:
# #                     key = f"Curve_{len(curves) + 1}"
# #                     curves[key] = current_curve
# #                     current_curve = []
# #     return curves

# # def plot_curves(curves):
# #     for key, curve in curves.items():
# #         iterations, losses = zip(*curve)
# #         plt.plot(iterations, losses, label=key)
# #     plt.title('Loss Curves')
# #     plt.xlabel('Iteration')
# #     plt.ylabel('Loss')
# #     plt.legend()
# #     plt.grid(True)
# #     plt.show()

# # if __name__ == "__main__":
# #     log_file = "/home/healthcrafters/healthcrafters/UA-MT/code/uamtbamlabel.out"  # Change this to the path of your log file
# #     curves = parse_log_file(log_file)
# #     plot_curves(curves)


# # # import matplotlib.pyplot as plt

# # # def parse_log_file(log_file):
# # #     iterations = []
# # #     losses = []
# # #     with open(log_file, 'r') as f:
# # #         for line in f:
# # #             if line.startswith('iteration'):
# # #                 parts = line.split(':')
# # #                 iteration = int(parts[0].split()[-1])
# # #                 print(iteration, parts[-1].strip())
# # #                 loss = float(parts[-1].strip().split()[0])
# # #                 iterations.append(iteration)
# # #                 losses.append(loss)
# # #     return iterations, losses

# # # def plot_loss_curve(iterations, losses):
# # #     plt.plot(iterations, losses, marker='o', linestyle='-')
# # #     plt.title('Loss Curve')
# # #     plt.xlabel('Iteration')
# # #     plt.ylabel('Loss')
# # #     plt.grid(True)
# # #     plt.show()

# # # if __name__ == "__main__":
# # #     log_file = "/home/healthcrafters/healthcrafters/UA-MT/code/newbamout.log"  # Change this to the path of your log file
# # #     iterations, losses = parse_log_file(log_file)
# # #     plot_loss_curve(iterations, losses)
