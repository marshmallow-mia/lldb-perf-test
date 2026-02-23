import subprocess
import pathlib
import shutil

# Function to resolve the server binary path
def resolve_server_binary(server):
    resolved_path = pathlib.Path(server).expanduser().resolve() if pathlib.Path(server).exists() else shutil.which(server)
    if resolved_path:
        return resolved_path
    else:
        raise FileNotFoundError(f"Error: {server} not found.")

# Function to get server version
def get_server_version(server):
    server_binary = resolve_server_binary(server)
    try:
        version_output = subprocess.check_output([server_binary, '--version'], stderr=subprocess.STDOUT)
        return version_output.strip()
    except subprocess.CalledProcessError as e:
        # If the subprocess returns a non-zero exit code, treat it as version not found
        return None, e.output

# Usage example
try:
    version = get_server_version('./llama-server')  # Using relative path
    print(f"Server version: {version}")
except Exception as e:
    print(e)
