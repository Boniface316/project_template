import subprocess
import sys


def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command '{command}': {e}")
        sys.exit(1)


def create_and_push_github_repo():
    project_name = "{{ cookiecutter.project_name }}"
    organization = input(
        "Enter the GitHub organization name (leave blank for personal account): "
    ).strip()
    visibility = (
        input("Should the repository be public or private? [public/private]: ")
        .strip()
        .lower()
    )

    if visibility not in ["public", "private"]:
        print("Invalid visibility option. Defaulting to public.")
        visibility = "public"

    repo_name = f"{organization}/{project_name}" if organization else project_name

    try:
        # Initialize local Git repository
        run_command("git init")
        run_command("git add .")
        run_command('git commit -m "Initial commit"')

        # Create GitHub repository
        subprocess.run(
            ["gh", "repo", "create", repo_name, f"--{visibility}"], check=True
        )
        print(f"Successfully created GitHub repository: {repo_name}")

        # Add remote and push
        remote_url = f"https://github.com/{repo_name}.git"
        run_command(f"git remote add origin {remote_url}")
        run_command("git push -u origin main")

        print(f"Successfully pushed local files to GitHub repository: {repo_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error during repository creation or push: {e}")
        sys.exit(1)


if __name__ == "__main__":
    create_and_push_github_repo()
