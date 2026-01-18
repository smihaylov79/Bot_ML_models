from utils import config
import importlib

def edit_config():
    print("\n=== Edit Config ===")

    # Collect all uppercase config variables
    settings = {k: getattr(config, k) for k in dir(config) if k.isupper()}

    # Display them
    for i, (key, value) in enumerate(settings.items(), start=1):
        print(f"{i}. {key:20} = {value}")

    # Choose which one to edit
    choice = input("\nSelect a setting to edit (number): ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(settings)):
        print("Invalid choice.")
        return

    key = list(settings.keys())[int(choice) - 1]
    old_value = settings[key]

    # Ask for new value
    new_value_raw = input(f"Enter new value for {key} (current: {old_value}): ").strip()

    # Convert type automatically
    try:
        if isinstance(old_value, int):
            new_value = int(new_value_raw)
        elif isinstance(old_value, float):
            new_value = float(new_value_raw)
        elif isinstance(old_value, bool):
            new_value = new_value_raw.lower() == "true"
        else:
            new_value = new_value_raw
    except:
        print("Invalid type.")
        return

    # Rewrite config.py
    with open("utils/config.py", "r") as f:
        lines = f.readlines()

    with open("utils/config.py", "w") as f:
        for line in lines:
            if line.startswith(key):
                f.write(f"{key} = {repr(new_value)}\n")
            else:
                f.write(line)

    print(f"{key} updated to {new_value}")

    # Reload config so changes take effect immediately
    importlib.reload(config)
