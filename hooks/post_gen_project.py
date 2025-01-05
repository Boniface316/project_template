import os
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
    def add_submodule(repo_name, path):
        try:
            print(f"Creating and adding submodule: {repo_name}")
            print(f"Path: {path}")
            os.mkdir(path)
            run_command(f"gh repo create {repo_name} --public")
            os.chdir(path)
            run_command("touch README.md")
            run_command("git add .")
            run_command('git commit -m "Initial commit"')
            run_command("git push -u origin main")
            os.chdir("..")
            run_command(
                f"git submodule add https://github.com/{github_username}/{repo_name}.git {path}"
            )

        except subprocess.CalledProcessError as e:
            if "already exists in the index" in e.stderr.decode():
                print(f"Submodule {path} already exists, skipping creation.")
            else:
                raise

    # Create notes repo and add as submodule
    notes_repo = f"{project_name}_notes"
    add_submodule(notes_repo, "notes")

    # Create benchmark repo and add as submodule
    benchmark_repo = f"{project_name}_benchmark"
    add_submodule(benchmark_repo, "benchmark")

    # Commit the submodules in the main repository
    run_command("git add .gitmodules notes benchmark")
    run_command('git commit -m "Add notes and benchmark submodules"')
    run_command("git push")

    print("Additional repositories created and added as submodules successfully.")


if __name__ == "__main__":
    create_and_push_github_repo()
