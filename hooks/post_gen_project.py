import os
import subprocess
import sys


def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command '{command}': {e}")
        sys.exit(1)


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

    github_username = get_github_username()
    if not github_username:
        print(
            "Failed to retrieve GitHub username. Make sure you're authenticated with gh CLI."
        )
        sys.exit(1)

    repo_name = (
        f"{organization}/{project_name}"
        if organization
        else f"{github_username}/{project_name}"
    )

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

        # Create additional repositories
        create_additional_repos(project_name, github_username)

    except subprocess.CalledProcessError as e:
        print(f"Error during repository creation or push: {e}")
        sys.exit(1)


def create_additional_repos(project_name, github_username):
    # Create notes repo and folder
    notes_repo = f"{project_name}_notes"
    run_command(f"gh repo create {notes_repo} --public")
    os.mkdir("notes")
    os.chdir("notes")
    run_command("git init")
    run_command(
        f"git remote add origin https://github.com/{github_username}/{notes_repo}.git"
    )
    run_command("git add .")
    run_command('git commit -m "Initial commit for notes"')
    run_command("git push -u origin main")
    os.chdir("..")

    # Create benchmark repo and folder
    benchmark_repo = f"{project_name}_benchmark"
    run_command(f"gh repo create {benchmark_repo} --public")
    os.mkdir("benchmark")
    os.chdir("benchmark")
    run_command("git init")
    run_command(
        f"git remote add origin https://github.com/{github_username}/{benchmark_repo}.git"
    )
    run_command("git add .")
    run_command('git commit -m "Initial commit for benchmark"')
    run_command("git push -u origin main")
    os.chdir("..")

    # Add notes and benchmark folders to .gitignore
    with open(".gitignore", "a") as gitignore:
        gitignore.write("\n# Additional folders\nnotes/\nbenchmark/\n")

    print("Additional repositories and folders created successfully.")


if __name__ == "__main__":
    create_and_push_github_repo()
