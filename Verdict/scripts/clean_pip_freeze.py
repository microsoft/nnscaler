#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import subprocess


def clean_pip_freeze():
    raw = subprocess.check_output(["pip", "freeze"]).decode().splitlines()
    cleaned = []

    for line in raw:
        if "@ file://" in line:
            pkg = line.split("@")[0].strip()
            try:
                metadata = subprocess.check_output(["pip", "show", pkg]).decode()
                version = next(
                    (
                        l.split(":")[1].strip()
                        for l in metadata.splitlines()
                        if l.startswith("Version:")
                    ),
                    None,
                )
                if version:
                    cleaned.append(f"{pkg}=={version}")
                else:
                    cleaned.append(f"# Could not find version for {pkg}")
            except subprocess.CalledProcessError:
                cleaned.append(f"# Error getting version for {pkg}")
        else:
            cleaned.append(line)

    with open("pip-requirements.txt", "w") as f:
        f.write("\n".join(cleaned) + "\n")

    print("✅ Cleaned requirements written to cleaned-requirements.txt")


clean_pip_freeze()
