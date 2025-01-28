import subprocess
import sys

def run_command(command):
    result = subprocess.run(
        command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout.decode().strip()


def get_github_username():
    try:
        result = subprocess.run(
            ["gh", "api", "user", "-q", ".login"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting GitHub username: {e}")
        return None


def create_and_push_github_repo():
    repo_name = "{{ cookiecutter.repository }}"
    organization = input(
        "Enter the GitHub organization name (leave blank for personal account): "
    ).strip()
    visibility = (
        input("Should the repository be public or private(default)? [public/private]: ")
        .strip()
        .lower()
    )

    if visibility not in ["public", "private"]:
        print("Invalid visibility option. Defaulting to private.")
        visibility = "private"

    github_username = get_github_username()
    if not github_username:
        print(
            "Failed to retrieve GitHub username. Make sure you're authenticated with gh CLI."
        )
        sys.exit(1)

    repo_name = (
        f"{organization}/{repo_name}"
        if organization
        else f"{github_username}/{repo_name}"
    )

    try:
        # Initialize local Git repository
        run_command("git init")
        run_command("git add .")
        run_command('git commit -m "Initial commit"')

        # Create GitHub repository
        run_command(f"gh repo create {repo_name} --{visibility}")
        print(f"Successfully created GitHub repository: {repo_name}")

        # Add remote and push
        remote_url = f"https://github.com/{repo_name}.git"
        run_command(f"git remote add origin {remote_url}")
        run_command("git push -u origin main")

        print(f"Successfully pushed local files to GitHub repository: {repo_name}")

    except Exception as e:
        print(f"Error during repository creation or push: {e}")
        sys.exit(1)


if __name__ == "__main__":
    repo_confirmation = input("Do you want to create and push the GitHub repository? [y/n]: ")
    if repo_confirmation.lower() not in ["y", "yes"]:
        sys.exit(0)
    print("Creating and pushing GitHub repository...")
    create_and_push_github_repo()
