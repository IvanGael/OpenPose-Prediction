import sys
import os

def run_visualization():
    if len(sys.argv) < 2:
        print("Usage: py main.py [visualization_script] [additional_arguments]")
        return

    script = sys.argv[1]
    script_path = os.path.join(script + ".py")

    if not os.path.isfile(script_path):
        print(f"Error: Visualization script '{script}' not found.")
        return

    # Remove the script name from the arguments
    args = sys.argv[2:]

    # Build the command
    command = ["py", script_path] + args

    # Execute the command
    os.system(" ".join(command))

if __name__ == "__main__":
    run_visualization()

