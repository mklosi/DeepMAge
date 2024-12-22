import re


def extract_unique_config_ids(log_files):
    # Regular expression to extract the config_id from the lines
    config_id_pattern = re.compile(r"--- Train pipeline for config_id: (\w+)")

    # Set to store unique config_ids
    config_ids = set()

    # Iterate over each log file
    for log_file in log_files:
        try:
            with open(log_file, 'r') as file:
                for line in file:
                    # Match the line against the pattern
                    match = config_id_pattern.search(line)
                    if match:
                        # Extract and store the config_id
                        config_ids.add(match.group(1))
        except FileNotFoundError:
            print(f"File not found: {log_file}")
        except Exception as e:
            print(f"Error processing {log_file}: {e}")

    # Convert the set to a sorted list
    sorted_config_ids = sorted(config_ids)

    # # Print the unique config_ids
    # print(sorted_config_ids)

    return sorted_config_ids


roise0r_log_files = [
    "/Users/martinklosi/Library/Mobile Documents/com~apple~CloudDocs/intellij-projects/DeepMAge/logs/process_1.log",
    "/Users/martinklosi/Library/Mobile Documents/com~apple~CloudDocs/intellij-projects/DeepMAge/logs/process_2.log",
    "/Users/martinklosi/Library/Mobile Documents/com~apple~CloudDocs/intellij-projects/DeepMAge/logs/process_3.log",
    "/Users/martinklosi/Library/Mobile Documents/com~apple~CloudDocs/intellij-projects/DeepMAge/logs/process_4.log",
    "/Users/martinklosi/Library/Mobile Documents/com~apple~CloudDocs/intellij-projects/DeepMAge/logs/process_5.log",
    "/Users/martinklosi/Library/Mobile Documents/com~apple~CloudDocs/intellij-projects/DeepMAge/logs/process_6.log",
    "/Users/martinklosi/Library/Mobile Documents/com~apple~CloudDocs/intellij-projects/DeepMAge/logs/process_7.log",
    "/Users/martinklosi/Library/Mobile Documents/com~apple~CloudDocs/intellij-projects/DeepMAge/logs/process_8.log",
]

mklosi_log_files = [
    "/Users/martinklosi/intellij-projects/DeepMAge/logs/process_1.log",
    "/Users/martinklosi/intellij-projects/DeepMAge/logs/process_2.log",
    "/Users/martinklosi/intellij-projects/DeepMAge/logs/process_3.log",
    "/Users/martinklosi/intellij-projects/DeepMAge/logs/process_4.log",
    "/Users/martinklosi/intellij-projects/DeepMAge/logs/process_5.log",
    "/Users/martinklosi/intellij-projects/DeepMAge/logs/process_6.log",
    "/Users/martinklosi/intellij-projects/DeepMAge/logs/process_7.log",
    "/Users/martinklosi/intellij-projects/DeepMAge/logs/process_8.log",
]

config_ids = extract_unique_config_ids(mklosi_log_files)

for config_id in config_ids:
    print(config_id)
