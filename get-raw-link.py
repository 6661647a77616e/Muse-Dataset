import requests

def fetch_repo_raw_paths(owner, repo, branch="main"):
    """
    Fetches the raw content paths for all files in a GitHub repository.

    Args:
        owner (str): GitHub username or organization name.
        repo (str): Repository name.
        branch (str): Branch name (default is 'main').

    Returns:
        list: List of raw content URLs.
    """
    base_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    raw_base_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"
    
    def get_file_paths(api_url):
        response = requests.get(api_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch repository contents: {response.json().get('message', 'Unknown error')}")
        contents = response.json()
        file_paths = []
        
        for item in contents:
            if item['type'] == 'file':
                file_paths.append(item['path'])
            elif item['type'] == 'dir':
                # Recursively fetch contents of subdirectories
                file_paths.extend(get_file_paths(item['url']))
        
        return file_paths

    file_paths = get_file_paths(base_api_url)
    raw_paths = [f"{raw_base_url}/{path}" for path in file_paths]
    return raw_paths

# Example usage:
if __name__ == "__main__":
    owner = "6661647a77616e"  # Replace with the username or organization name
    repo = "Muse-Dataset"  # Replace with the repository name
    branch = "main"  # Replace with the branch name if not 'main'

    try:
        raw_paths = fetch_repo_raw_paths(owner, repo, branch)
        print("Raw Content Paths:")
        for path in raw_paths:
            print(path)
    except Exception as e:
        print(f"Error: {e}")
